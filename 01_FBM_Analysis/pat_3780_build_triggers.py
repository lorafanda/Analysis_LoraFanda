#!/usr/bin/env python3
"""
pat_3780_build_triggers.py - PAT_3780's trial triggers, from the photodiode hidden in X1 / X2.

    python pat_3780_build_triggers.py            # write the three prep0 tables, archive the 2025 ones
    python pat_3780_build_triggers.py --check    # detect, match and report; write nothing
    python pat_3780_build_triggers.py --plot     # also draw the diagnostic PNG
    python pat_3780_build_triggers.py --out DIR  # write everything to DIR instead of prep0 (a test)

THE RECORDING. One Micromed file, EEG_2601802.TRC (11:00:04, 1224 s, 152 channels at
2048 Hz). The task sits inside it: the stimulus log's clock maps onto the recording with
a constant of about +149.6 s, which the script fits, not assumes.

THE PHOTODIODE IS AN ENVELOPE DIFFERENCE. X1 and X2 carry the same 50-Hz carrier, driven
into the +-3200 uV rails 94 % of the time, so X1, X2, X1 - X2 and every other pair of the
X inputs show nothing at the trials (checked, 2026-10-07). The diode amplitude-modulates
that carrier: take the Hilbert envelope of each input, subtract, low-pass at 20 Hz, and
every screen change is a step of 10-20 uV against a floor of ~2 uV (MAD) that decays back
to zero within a few seconds (AC-coupled, like PAT_1327's). The difference is flat outside
the task. DOWN IS STIMULUS ON, UP IS STIMULUS OFF: down edge then up edge, and the interval
between them is the stimulus duration - 1.01 s for picture naming, 3.51 s for reading
completion, 2-4.4 s for the auditory words, the task's own constants.

THE 2025 TABLES WERE 180.0 s TOO EARLY. prep0/PAT_3780_LM_*__X2_2025-10-06.tsv came from
PAT_3780_FLM_all.tsv, which is the stimulus log minus 30.42 s; the diode says plus 149.6 s.
Nothing in the EEG was locked to the old onsets at any lag within +-150 s; at the diode's
onsets the superior temporal contacts (TOG1-5) answer the spoken words with |t| above 10
at ~210 ms and 43 contacts answer the pictures. Those tables are moved to prep0/archive
when this script writes (140's collect_trials reads EVERY .tsv in prep0, so leaving them
would double the trials).

MATCHING. The constant is the lag that lands the most log onsets within 50 ms of a down
edge. Each log trial then takes the nearest down edge within MATCH_WIN_S and the first up
edge after it. A stimulus duration that does not fit its block (an up edge missed or
borrowed from the next trial) falls back to the design duration of that trial, taken from
the 2025 table by trial_idx, and is named in the report. Against the log the diode edges
scatter by +-10-20 ms per trial, with a few picture trials where the screen was 50-160 ms
late: the diode is the truth, the log is not frame-accurate.

trial_end follows the convention the 2025 tables and the rest of the HUG cohort use:
stimulus offset + the log's `duration` (the response window), which puts it a constant
~1.1-1.4 s before the next onset. Written: onset, onset_duration, sample, sample_offsets,
trial_end, condition_name, resp_accuracy, trial_idx, plus `source` (photodiode | design).
"""
from __future__ import print_function

import argparse
import datetime as _dt
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "functions"))
import config as cfg                                                  # noqa: E402

PID = "PAT_3780"
RAW = os.path.join(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_HUG",
                   PID, "task_FBM", "data_LM", "raw")
PREP0 = os.path.join(os.path.dirname(RAW), "prep0")
TRC = "EEG_2601802.TRC"
EVENTS = "sub-PAT3780_task-LanguageMapping_timestamp-15-5-2025(10h59m4s)_lang-FRE_events.tsv"
OLD_TABLES = "%s_LM_*__X2_2025-10-06.tsv" % PID        # the 180-s-early tables, archived on write

PD_CHANNELS = ("X1", "X2")       # the diode is the difference of the ENVELOPES of these two
LOWPASS_HZ = 20.0                # on the envelope difference
THR_MAD = 4.0                    # edge threshold in MADs of the difference (about 7 uV here)
MIN_GAP_S = 0.30                 # two edges of the same sign cannot be closer than this
MATCH_WIN_S = 0.35               # how far a log onset may sit from its down edge
DUR_TOL = 0.25                   # a stimulus duration may differ from the block design by this fraction

# the events table's category -> (the pipeline's condition_name, the prep0 file's tag)
COND = {
    "picture_naming":      ("picture_naming", "picture"),
    "auditory_naming_FRE": ("auditory_naming_FRE", "audio"),
    "reading_completion":  ("reading_completion", "reading"),
}


def read_pd():
    """envelope(X1) - envelope(X2), low-passed, plus fs."""
    from neo.rawio import MicromedRawIO
    from scipy.signal import butter, hilbert, sosfiltfilt
    path = os.path.join(RAW, TRC)
    r = MicromedRawIO(filename=path)
    r.parse_header()
    chans = [str(c["name"]).strip() for c in r.header["signal_channels"]]
    missing = [c for c in PD_CHANNELS if c not in chans]
    if missing:
        raise SystemExit("%s: no channel %s" % (TRC, ", ".join(missing)))
    fs = float(r.get_signal_sampling_rate())
    n = int(r.get_signal_size(block_index=0, seg_index=0))
    env = []
    for c in PD_CHANNELS:
        idx = np.array([chans.index(c)])
        raw = r.get_analogsignal_chunk(block_index=0, seg_index=0, i_start=0, i_stop=n, channel_indexes=idx)
        x = r.rescale_signal_raw_to_float(raw, dtype="float32", channel_indexes=idx)[:, 0].astype(np.float64)
        rails = float(np.mean(np.abs(x) >= 0.999 * np.abs(x).max()))
        print("  %-3s %10s samples, %4.0f %% at the rails" % (c, "{:,}".format(n), 100 * rails))
        env.append(np.abs(hilbert(x - np.median(x))))
    d = env[0] - env[1]
    sos = butter(4, LOWPASS_HZ / (fs / 2.0), "low", output="sos")
    return sosfiltfilt(sos, d), fs


def edges(d, fs):
    """(down_samples, up_samples): negative- and positive-going crossings of +-threshold."""
    mad = 1.4826 * np.median(np.abs(d - np.median(d)))
    thr = THR_MAD * mad
    down = np.flatnonzero((d[1:] < -thr) & (d[:-1] >= -thr)) + 1
    up = np.flatnonzero((d[1:] > thr) & (d[:-1] <= thr)) + 1

    def thin(ev):
        keep = [0]
        for j in range(1, len(ev)):
            if (ev[j] - ev[keep[-1]]) / fs >= MIN_GAP_S:
                keep.append(j)
        return ev[keep]
    if down.size == 0 or up.size == 0:
        raise SystemExit("no photodiode edges above %.1f uV - wrong channels?" % thr)
    return thin(down), thin(up), thr, mad


def design_durations():
    """trial_idx -> stimulus duration of the 2025 tables (the task's design values)."""
    out = {}
    for p in glob.glob(os.path.join(PREP0, OLD_TABLES)) + glob.glob(os.path.join(PREP0, "archive", OLD_TABLES)):
        t = pd.read_csv(p, sep="\t")
        tag = os.path.basename(p).split("__")[0].split("_LM_")[1]
        for _, r in t.iterrows():
            out[(tag, int(r.trial_idx))] = float(r.onset_duration)
    return out


def build(check=False, plot=False, out_dir=None):
    out_dir = out_dir or PREP0
    print("[pd] reading %s and %s from %s" % (PD_CHANNELS + (TRC,)))
    d, fs = read_pd()
    print("  {:,} samples @ {:g} Hz = {:.1f} min".format(len(d), fs, len(d) / fs / 60))
    dn_i, up_i, thr, mad = edges(d, fs)
    dn_t, up_t = dn_i / fs, up_i / fs
    print("[pd] threshold %.1f uV (%.0f x MAD %.2f): %d down edges, %d up edges, %.1f -> %.1f min"
          % (thr, THR_MAD, mad, len(dn_i), len(up_i), dn_t[0] / 60, max(dn_t[-1], up_t[-1]) / 60))

    log = pd.read_csv(os.path.join(RAW, EVENTS), sep="\t")
    log = log[log.category.isin(COND)].sort_values("onset").reset_index(drop=True)
    print("[log] %d trials: %s" % (len(log), dict(log.category.value_counts())))

    # the constant that puts the log on the recording clock: scanned over the down edges
    best = (None, -1)
    for cand in dn_t - log.onset.values[0]:
        want = log.onset.values + cand
        j = np.clip(np.searchsorted(dn_t, want), 1, len(dn_t) - 1)
        res = np.minimum(np.abs(dn_t[j] - want), np.abs(dn_t[j - 1] - want))
        hits = int((res < 0.05).sum())
        if hits > best[1]:
            best = (cand, hits)
    lag = best[0]
    print("[fit] log -> recording: +%.3f s (%d/%d onsets within 50 ms of a down edge)" % (lag, best[1], len(log)))
    old_const = None
    old = glob.glob(os.path.join(PREP0, OLD_TABLES)) + glob.glob(os.path.join(PREP0, "archive", OLD_TABLES))
    if old:
        t_old = pd.read_csv(old[0], sep="\t")
        m = t_old.merge(log, left_on="trial_idx", right_on="exemplar")
        if len(m):
            old_const = float(np.median(m.onset_x - m.onset_y))
            print("[fit] the 2025 tables used %.3f s: they were %.1f s early" % (old_const, lag - old_const))

    design = design_durations()
    rows, unmatched, fixed = [], [], []
    for i, r in log.iterrows():
        want = r.onset + lag
        j = int(np.argmin(np.abs(dn_t - want)))
        if abs(dn_t[j] - want) > MATCH_WIN_S:
            unmatched.append((r.category, int(r.exemplar), round(float(r.onset + lag), 2)))
            continue
        on = int(dn_i[j])
        cond_name, tag = COND[r.category]
        ref = design.get((tag, int(r.exemplar)))
        source = "photodiode"
        if ref is not None:
            # the up edge nearest to where the design says the stimulus ends. The difference
            # keeps oscillating while a sound plays, so "the first up edge after the onset"
            # would stop 16 of the 50 auditory trials after a few hundred ms.
            want_off = on + ref * fs
            k = int(np.argmin(np.abs(up_i - want_off)))
            off = int(up_i[k])
            if abs(off - want_off) / fs > MATCH_WIN_S:
                fixed.append("%s %d: no up edge within %.2f s of the design offset (%.2f s), nearest is %+.2f s"
                             % (tag, int(r.exemplar), MATCH_WIN_S, ref, (off - want_off) / fs))
                off = int(round(want_off))
                source = "design"
        else:
            k = int(np.searchsorted(up_i, on))
            if k >= len(up_i):
                unmatched.append((r.category, int(r.exemplar), round(float(want), 2)))
                continue
            off = int(up_i[k])
        rows.append(dict(sample=on, sample_offsets=off, condition_name=cond_name, tag=tag,
                         resp_accuracy=str(r.response_type).strip(), trial_idx=int(r.exemplar),
                         log_duration=float(r.duration), res_s=dn_t[j] - want, source=source))
    tr = pd.DataFrame(rows).sort_values("sample").reset_index(drop=True)
    if unmatched:
        print("[warn] %d log trials with no photodiode edge within %.2f s: %s" % (len(unmatched), MATCH_WIN_S, unmatched))
    for f in fixed:
        print("[fix]  %s -> offset set from the design" % f)
    if tr["sample"].duplicated().any():
        raise SystemExit("the same down edge was used for two trials - matching is wrong")

    tr["onset"] = tr["sample"] / fs
    tr["onset_duration"] = (tr.sample_offsets - tr["sample"]) / fs
    tr["trial_end"] = (tr.sample_offsets + np.round(tr.log_duration * fs)).astype(np.int64)
    res_ms = tr.res_s * 1000
    print("[chk] diode onset - (log onset + lag): median %+.0f ms, IQR %+.0f..%+.0f, range %+.0f..%+.0f ms"
          % (np.median(res_ms), np.percentile(res_ms, 25), np.percentile(res_ms, 75), res_ms.min(), res_ms.max()))
    sl = np.polyfit(tr.onset, tr.res_s, 1)[0]
    print("[chk] drift of that residual over the recording: %+.1f ms per 1000 s" % (sl * 1e6))
    print("[chk] stimulus duration per block (the task's own constants):")
    for c, g in tr.groupby("condition_name"):
        print("        %-20s n=%3d  min %.2f  med %.2f  max %.2f  sd %.3f s  (%d from the design)"
              % (c, len(g), g.onset_duration.min(), g.onset_duration.median(), g.onset_duration.max(),
                 g.onset_duration.std(), int((g.source == "design").sum())))
    nxt = tr["sample"].shift(-1)
    iti = ((nxt - tr.trial_end) / fs)[tr.condition_name.values == tr.condition_name.shift(-1).values]
    print("[chk] trial_end -> next onset inside a block: min %.2f  med %.2f  max %.2f s" % (iti.min(), iti.median(), iti.max()))
    post_s = (tr.trial_end - tr.sample_offsets) / fs
    print("[chk] post-stimulus window: min %.2f  med %.2f  max %.2f s   (the filters keep %.1f-%g s)"
          % (post_s.min(), post_s.median(), post_s.max(), cfg.min_post_s, cfg.max_post_s))
    keep = ((tr.onset_duration >= cfg.min_stim_s) & (post_s >= cfg.min_post_s) &
            (post_s <= cfg.max_post_s) & (tr.resp_accuracy.str.lower() == "correct"))
    print("[chk] would survive 140's hard filters (before the IQR rule): %d of %d" % (int(keep.sum()), len(tr)))
    for c, g in tr.assign(keep=keep).groupby("condition_name"):
        print("        %-20s %d of %d" % (c, int(g.keep.sum()), len(g)))

    cols = ["onset", "onset_duration", "sample", "sample_offsets", "trial_end",
            "condition_name", "resp_accuracy", "trial_idx", "source"]
    stamp = _dt.date.today().isoformat()
    if check:
        print("\n--check: nothing written")
        return tr
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if os.path.normcase(os.path.abspath(out_dir)) == os.path.normcase(os.path.abspath(PREP0)):
        arch = os.path.join(PREP0, "archive")
        if not os.path.isdir(arch):
            os.makedirs(arch)
        for p in glob.glob(os.path.join(PREP0, OLD_TABLES)):
            dst = os.path.join(arch, os.path.basename(p))
            if os.path.exists(dst):
                dst = dst[:-4] + "_superseded-%s.tsv" % stamp
            shutil.move(p, dst)
            print("archived %s -> %s" % (os.path.basename(p), os.path.relpath(dst, PREP0)))
    for tag, g in tr.groupby("tag"):
        p = os.path.join(out_dir, "%s_LM_%s__X1X2env_%s.tsv" % (PID, tag, stamp))
        body = g[cols].sort_values("sample").to_csv(sep="\t", index=False)
        with open(p, "w", newline="") as fh:                 # write, then read back: the share drops bytes
            fh.write(body)
        with open(p, "r", newline="") as fh:
            if fh.read() != body:
                raise SystemExit("%s came back different from what was written - the share truncated it; re-run" % p)
        print("wrote %s  (%d trials)" % (p, len(g)))
    if plot:
        _plot(d, fs, tr, out_dir)
    return tr


def _plot(d, fs, tr, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = ["picture", "audio", "reading"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 9))
    for j, tag in enumerate(tags):
        g = tr[tr.tag == tag].sort_values("onset_duration").reset_index(drop=True)
        if not len(g):
            continue
        w0, w1 = int(1.0 * fs), int(g.onset_duration.max() * fs + 2.0 * fs)
        M = np.array([d[s - w0:s + w1] for s in g["sample"]])
        tt = np.arange(-w0, w1) / fs
        axes[0, j].imshow(M, aspect="auto", extent=[tt[0], tt[-1], len(g) - 0.5, -0.5], cmap="RdBu_r", vmin=-15, vmax=15)
        axes[0, j].axvline(0, color="k", lw=1)
        axes[0, j].plot(g.onset_duration, np.arange(len(g)), "|", color="m", ms=6, mew=1.3)
        axes[0, j].set_title("%s: envelope(X1) - envelope(X2) around the onsets (magenta = offsets)" % tag, fontsize=9, loc="left")
        axes[0, j].set_xlabel("s from onset")
        axes[1, j].plot(g.onset, g.res_s * 1000, "o", ms=4)
        axes[1, j].axhline(0, color="k", lw=.6)
        axes[1, j].set_title("%s: diode onset minus (log onset + lag)" % tag, fontsize=9, loc="left")
        axes[1, j].set_xlabel("recording time (s)"); axes[1, j].set_ylabel("ms")
    fig.suptitle("%s - photodiode as the envelope difference of X1 and X2" % PID, x=.02, ha="left")
    fig.tight_layout()
    p = os.path.join(out_dir, "%s_LM_photodiode_check.png" % PID)
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print("plot  %s" % p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true", help="report, write nothing")
    ap.add_argument("--plot", action="store_true", help="also write the diagnostic PNG")
    ap.add_argument("--out", default=None, help="write to this folder instead of prep0 (no archiving)")
    a = ap.parse_args()
    build(check=a.check, plot=a.plot, out_dir=a.out)
