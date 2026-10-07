#!/usr/bin/env python3
"""
el031_build_triggers.py - EL031's trial triggers, from three recordings and three logs.

    python el031_build_triggers.py            # write the two files and the photodiode figure
    python el031_build_triggers.py --check    # detect, match and report, write nothing
    python el031_build_triggers.py --out DIR  # write everything to DIR instead (a test)

EL031's language mapping (Bern, March 2024) was run as three separate recordings, one
block each, logged in three events tables:

    EL031_20240319_HUG_picturesLM.EDF   19 Mar 09:29, 912 s   picture naming
    EL031_20240319_HUG_audioLM.EDF      19 Mar 10:45, 846 s   auditory naming, stopped
                                                               after 23 trials (patient asleep)
    EL031_20240320_HUG_sentenceLM.EDF   20 Mar 14:22, 846 s   reading completion

The pipeline wants one recording and one events table per patient. cfg.RAW_CONCAT joins
the three files in that order; the 20 March file calls the first shaft antSUP where the
other two and the Lookup workbook say antSFG, which cfg.RAW_CONCAT_RENAME folds back.
This script reads the photodiode (DC6) of each file on its own - the DC baselines differ
between files, so one threshold over the joined trace would not do - fits the constant
that puts each log onto its recording (the lag that lands the most log onsets within
100 ms of a pulse), pairs every log row with its pulse, and keeps only the trials that
belong to the block of that recording:

  * the picture session: 54 one-second pulses; the first three log rows are the training
    trials (two have no pulse, the third a 42-s response) and are dropped - 51 trials
    (Camille's notes: "photodiode missing for the 3 first training trials, not for the
    true trials"); two one-second pulses before the run belong to no log row;
  * the audio session: 23 auditory pulses of 3.2-6.8 s, 20 correct; the 6 picture rows
    of its log are strays (an aborted picture run inside the audio session) and go;
  * the sentence session: 53 reading pulses of 3.50 s; the 3 picture rows and the 1
    audio row at the start of its log are strays and go.

Written:
    data_LM/prep0/EL031_LM_manual_trigs_concat.tsv
        sample, sample_offsets on the JOINED axis, 127 rows: 51 picture, 23 audio,
        53 reading, in that order. cfg.EL_PRESETS["EL031"]["manual_trig"] points at it,
        so 140 --pd bypasses its own detection.
    data_LM/raw/sub-EL031_task-LanguageMapping_merged_events.tsv
        the matching 127 log rows in the same order, Bern format, response_time named
        response_timex, each log's `onset` moved into the first log's frame (one constant
        per log, measured as pulse-minus-onset) so the PD-extraction plot's single shift
        holds for all three blocks.
    EL031_LM_photodiode_check.png
        per session: the whole DC6 trace with the kept pulses (green) and the ignored
        ones (grey), the raster of DC6 around every kept onset, and the per-trial residual
        of the pulse against the log. Saved beside the tables in prep0 and in the 03_ERSP
        tree under EL031/LM/PerTrial/Report (created if 140 has not run yet).

Nothing here touches the original logs or the recordings. The EDF reader below is a
plain header + int16 reader, so this runs without mne (the laptop) as well as with it.
"""
from __future__ import print_function

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "functions"))
import config as cfg                                                  # noqa: E402

PID = "EL031"
D = os.path.join(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_BERN", PID, "task_FBM", "data_LM")
RAW, PREP0 = os.path.join(D, "raw"), os.path.join(D, "prep0")
REPORT_DIR = os.path.join(cfg.outputs_root, getattr(cfg, "ERSP_TREE", "03_ERSP"), PID, "LM", "PerTrial", "Report")
TRIG = "DC6"
# file -> (its block's log category, the log file pattern, the pipeline's trial id)
SESSIONS = [
    ("EL031_20240319_HUG_picturesLM.EDF", "picture_naming",      "*events_picture.tsv",  "picture"),
    ("EL031_20240319_HUG_audioLM.EDF",    "auditory_naming_GER", "*events_auditory.tsv", "auditory"),
    ("EL031_20240320_HUG_sentenceLM.EDF", "reading_completion",  "*events_sentence.tsv", "reading"),
]
TRAINING_ROWS = {"picture_naming": 3}    # the first rows of that log that are training trials
MATCH_MS = 100.0                         # a log row needs a pulse this close (after the lag)
OUT_TRIGS = os.path.join(PREP0, "%s_LM_manual_trigs_concat.tsv" % PID)
OUT_EVENTS = os.path.join(RAW, "sub-%s_task-LanguageMapping_merged_events.tsv" % PID)


def edf_channel(path, want):
    """One channel of an EDF file as (signal in physical units, its sampling rate, n_samples
    of the file at the EEG rate). Plain reader: 256-byte header, 256 bytes per signal, then
    int16 records; signals may have different rates, each keeps its own."""
    with open(path, "rb") as f:
        h = f.read(256)
        n_rec, rec_dur, ns, hdr = int(h[236:244]), float(h[244:252]), int(h[252:256]), int(h[184:192])
        hh = f.read(256 * ns)
        labels = [hh[16 * i:16 * (i + 1)].decode().strip() for i in range(ns)]
        off = 16 * ns + 80 * ns + 8 * ns
        pmin = np.array([float(hh[off + 8 * i:off + 8 * (i + 1)]) for i in range(ns)]); off += 8 * ns
        pmax = np.array([float(hh[off + 8 * i:off + 8 * (i + 1)]) for i in range(ns)]); off += 8 * ns
        dmin = np.array([int(hh[off + 8 * i:off + 8 * (i + 1)]) for i in range(ns)]); off += 8 * ns
        dmax = np.array([int(hh[off + 8 * i:off + 8 * (i + 1)]) for i in range(ns)]); off += 8 * ns + 80 * ns
        nsamp = np.array([int(hh[off + 8 * i:off + 8 * (i + 1)]) for i in range(ns)])
        if want not in labels:
            raise SystemExit("%s: no channel %s" % (os.path.basename(path), want))
        ci = labels.index(want)
        rec_len, start = int(nsamp.sum()), int(nsamp[:ci].sum())
        f.seek(hdr)
        data = np.fromfile(f, dtype="<i2", count=n_rec * rec_len).reshape(n_rec, rec_len)
    x = data[:, start:start + nsamp[ci]].reshape(-1).astype(np.float64)
    x = (x - dmin[ci]) * (pmax[ci] - pmin[ci]) / (dmax[ci] - dmin[ci]) + pmin[ci]
    fs = nsamp[ci] / rec_dur
    n_eeg = int(nsamp[0] * n_rec)                     # the first signal is a depth contact: the EEG rate
    return x, fs, n_eeg, labels


def pulses(x, fs):
    """Photodiode pulses of one file: (onset_sample, offset_sample). The threshold sits
    halfway between the low state (30th percentile) and the pulse level (99.5th); pulses
    shorter than 0.2 s are switching noise and dropped. The same rule as EL051's."""
    xs = np.convolve(x, np.ones(10) / 10, mode="same")
    lo, hi = np.percentile(xs, [30, 99.5]); thr = (lo + hi) / 2
    h = xs > thr
    on = np.flatnonzero(h[1:] & ~h[:-1]) + 1
    off = np.flatnonzero(~h[1:] & h[:-1]) + 1
    if len(off) and len(on) and off[0] < on[0]:
        off = off[1:]
    n = min(len(on), len(off)); on, off = on[:n], off[:n]
    keep = (off - on) / fs > 0.2
    return on[keep], off[keep]


def read_log(path):
    return pd.read_csv(path, sep="\t", skiprows=1, dtype=str)


def fit_lag(pulse_t, onsets):
    """The constant that lands the most log onsets within MATCH_MS of a pulse."""
    best = (None, -1)
    for cand in (pulse_t[:, None] - onsets[None, :]).ravel():
        j = np.clip(np.searchsorted(pulse_t, onsets + cand), 1, len(pulse_t) - 1)
        res = np.minimum(np.abs(pulse_t[j] - (onsets + cand)), np.abs(pulse_t[j - 1] - (onsets + cand)))
        hits = int((res < MATCH_MS / 1000).sum())
        if hits > best[1]:
            best = (cand, hits)
    return best


def build(check=False, out_dir=None):
    concat = (getattr(cfg, "RAW_CONCAT", {}) or {}).get(PID)
    files = [s[0] for s in SESSIONS]
    if concat != files:
        raise SystemExit("cfg.RAW_CONCAT['%s'] must be %s, in this order (it is %s)" % (PID, files, concat))
    rows, figs, offset_axis, consts = [], [], 0, []
    for fn, block_cat, log_pat, trial_id in SESSIONS:
        x, fs, n_eeg, labels = edf_channel(os.path.join(RAW, fn), TRIG)
        on, off = pulses(x, fs)
        t, w = on / fs, (off - on) / fs
        log_path = glob.glob(os.path.join(RAW, log_pat))
        if len(log_path) != 1:
            raise SystemExit("%s: expected one log matching %s, found %s" % (fn, log_pat, log_path))
        L = read_log(log_path[0]).reset_index(drop=True)
        onsets = pd.to_numeric(L.onset).to_numpy()
        lag, hits = fit_lag(t, onsets)
        want = onsets + lag
        j = np.argmin(np.abs(t[:, None] - want[None, :]), axis=0)
        res = t[j] - want
        matched = np.abs(res) < MATCH_MS / 1000
        real = (L.category == block_cat).to_numpy()
        train = np.zeros(len(L), bool); train[:TRAINING_ROWS.get(block_cat, 0)] = True
        keep = matched & real & ~train
        n_tr = int(train.sum()); n_stray = int((~real).sum()); n_unm = int((real & ~train & ~matched).sum())
        print("[%s] %s" % (trial_id, fn))
        print("   %d pulses (%.1f-%.1f s of %.0f s); log %d rows %s" % (len(t), t[0], t[-1], x.size / fs, len(L), dict(L.category.value_counts())))
        print("   lag log -> recording %+.3f s: %d/%d rows within %d ms; kept %d %s trials (%d correct); dropped %d training, %d strays, %d unmatched"
              % (lag, hits, len(L), MATCH_MS, int(keep.sum()), trial_id, int((L.response_type[keep].str.strip() == "correct").sum()), n_tr, n_stray, n_unm))
        if n_unm:
            print("   unmatched %s rows (no pulse within %d ms): %s" % (block_cat, MATCH_MS, [(int(i), round(float(want[i]), 1)) for i in np.flatnonzero(real & ~train & ~matched)]))
        kw = w[j[keep]]
        print("   kept stimulus widths %.2f-%.2f s (log durations %.2f-%.2f s); residual pulse - log: median %+.0f ms, max |%.0f| ms; trial spacing median %.1f s"
              % (kw.min(), kw.max(), pd.to_numeric(L.duration[keep]).min(), pd.to_numeric(L.duration[keep]).max(),
                 np.median(res[keep]) * 1000, np.abs(res[keep]).max() * 1000, np.median(np.diff(t[j[keep]]))))
        # the depth channels must be the same set in every file, up to the rename
        ren = (getattr(cfg, "RAW_CONCAT_RENAME", {}) or {}).get(PID, {})
        depth = sorted(_rename(l, ren) for l in labels if any(c.isdigit() for c in l) and not l.upper().startswith(("DC", "EKG", "EOG", "F", "C", "P", "T", "O")))
        figs.append(dict(trial_id=trial_id, x=x, fs=fs, t=t, w=w, keep_idx=j[keep], res=res[keep], want=want[keep], depth=depth))
        for i in np.flatnonzero(keep):
            rows.append(dict(sample=int(on[j[i]]) + offset_axis, sample_offsets=int(off[j[i]]) + offset_axis, trial_id=trial_id, file=fn, log=L.iloc[i],
                             lag=lag + offset_axis / fs))          # log onset -> JOINED-axis time
        consts.append((trial_id, lag))
        print("   joined axis: this file starts at sample %d (%.1f s)" % (offset_axis, offset_axis / fs))
        offset_axis += n_eeg
    if len({tuple(f["depth"]) for f in figs}) != 1:
        raise SystemExit("the depth channels differ between the files even after RAW_CONCAT_RENAME: %s" % [f["depth"][:3] for f in figs])
    trig = pd.DataFrame({"sample": [r["sample"] for r in rows], "sample_offsets": [r["sample_offsets"] for r in rows]})
    ev = pd.DataFrame([r["log"] for r in rows]).reset_index(drop=True)
    ev = ev.rename(columns={"response_time": "response_timex"})
    # one time frame for `onset`: every block's rows are shifted so that pulse (on the joined
    # axis) minus onset is the constant of the first block, as the PD-extraction plot draws
    # the log with a single shift
    lag0 = consts[0][1]
    onset = pd.to_numeric(ev["onset"]).to_numpy(); lags = np.array([r["lag"] for r in rows])
    ev["onset"] = ["%.6f" % v for v in onset + (lags - lag0)]
    chk = trig["sample"].to_numpy() / fs - pd.to_numeric(ev["onset"]).to_numpy()
    if np.ptp(chk) > 0.15:
        raise SystemExit("pulse minus merged onset is not one constant (spread %.3f s)" % np.ptp(chk))
    print("merged log: pulse minus onset = %.3f s for every row (spread %.0f ms)" % (np.median(chk), np.ptp(chk) * 1000))
    ids = [r["trial_id"] for r in rows]
    counts = {k: ids.count(k) for k in dict.fromkeys(ids)}
    print("\n%d trials on the joined axis: %s; last trigger at %.1f s of %.1f s" % (len(rows), counts, trig.sample_offsets.max() / fs, offset_axis / fs))
    preset = (getattr(cfg, "EL_PRESETS", {}) or {}).get(PID, {}).get("trial_ids")
    if preset is not None and list(preset) != ids:
        print("[warn] cfg.EL_PRESETS['%s']['trial_ids'] does not match: the preset has %s, the recordings give %s - fix the preset before 140 --pd"
              % (PID, {k: list(preset).count(k) for k in dict.fromkeys(preset)}, counts))
    if check:
        print("(--check: nothing written)")
        return trig, ev
    out_trigs, out_events = OUT_TRIGS, OUT_EVENTS
    fig_dirs = [PREP0, REPORT_DIR]
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_trigs, out_events = os.path.join(out_dir, os.path.basename(OUT_TRIGS)), os.path.join(out_dir, os.path.basename(OUT_EVENTS))
        fig_dirs = [out_dir]
    os.makedirs(os.path.dirname(out_trigs), exist_ok=True)
    _write_verified(out_trigs, trig.to_csv(sep="\t", index=False, lineterminator="\n"))
    _write_verified(out_events, "SubjectNumber : %s\n" % PID + ev.to_csv(sep="\t", index=False, lineterminator="\n"))
    print("wrote %s  (%d rows)" % (out_trigs, len(trig)))
    print("wrote %s  (%d rows)" % (out_events, len(ev)))
    for d in fig_dirs:
        os.makedirs(d, exist_ok=True)
        _plot(figs, os.path.join(d, "%s_LM_photodiode_check.png" % PID))
    return trig, ev


def _rename(label, ren):
    for old, new in ren.items():
        if label.startswith(old) and label[len(old):].isdigit():
            return new + label[len(old):]
    return label


def _write_verified(path, body):
    with open(path, "w", encoding="utf-8", newline="") as f:      # write, read back: the share drops bytes
        f.write(body)
    with open(path, "r", encoding="utf-8", newline="") as f:
        if f.read() != body:
            raise SystemExit("%s came back different from what was written - the share truncated it; re-run" % path)


def _plot(figs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(figs), 3, figsize=(22, 4.2 * len(figs)), gridspec_kw=dict(width_ratios=[2.2, 1.3, 1]))
    for k, F in enumerate(figs):
        x, fs, t, w = F["x"], F["fs"], F["t"], F["w"]
        dec = max(1, int(fs // 64)); tt = np.arange(0, x.size, dec) / fs
        a = axes[k, 0]; a.plot(tt, x[::dec], color="k", lw=.4)
        kept = set(F["keep_idx"].tolist())
        for i in range(len(t)):
            a.axvspan(t[i], t[i] + w[i], color="#2a9d8f" if i in kept else "0.6", alpha=.45 if i in kept else .35, lw=0)
        a.set_xlim(0, x.size / fs); a.set_title("%s session: %s over the whole file; green = kept trials (%d), grey = ignored pulses (%d)"
                                                % (F["trial_id"], TRIG, len(kept), len(t) - len(kept)), fontsize=9, loc="left"); a.set_xlabel("s")
        on_s = t[F["keep_idx"]]; dur = w[F["keep_idx"]]
        w0, w1 = int(1.0 * fs), int(dur.max() * fs + 2.0 * fs)
        M = np.full((len(on_s), w0 + w1), np.nan)
        for i, s in enumerate(on_s):
            s0 = int(s * fs) - w0; seg = x[max(s0, 0):s0 + w0 + w1]; M[i, max(0, -s0):max(0, -s0) + seg.size] = seg
        tt2 = np.arange(-w0, w1) / fs; o = np.argsort(dur)
        b = axes[k, 1]; b.imshow(M[o], aspect="auto", extent=[tt2[0], tt2[-1], len(on_s) - 0.5, -0.5], cmap="Greys")
        b.axvline(0, color="#c1121f", lw=1); b.plot(dur[o], np.arange(len(on_s)), "|", color="m", ms=6, mew=1.2)
        b.set_title("%s around every kept onset (magenta = pulse end)" % TRIG, fontsize=9, loc="left"); b.set_xlabel("s from onset"); b.set_ylabel("trial, by stimulus duration")
        c = axes[k, 2]; c.plot(F["want"], F["res"] * 1000, "o", ms=4); c.axhline(0, color="k", lw=.6)
        c.set_title("pulse minus (log onset + lag), per trial", fontsize=9, loc="left"); c.set_xlabel("recording time (s)"); c.set_ylabel("ms")
    fig.suptitle("%s - photodiode %s of the three sessions" % (PID, TRIG), x=.02, ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    print("plot  %s" % path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true", help="detect and report, write nothing")
    ap.add_argument("--out", default=None, help="write the files and the figure to this folder instead (a test)")
    a = ap.parse_args()
    build(check=a.check, out_dir=a.out)
