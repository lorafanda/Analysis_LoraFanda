#!/usr/bin/env python3
"""
el051_build_triggers.py - EL051's trial triggers, from two recordings and two logs.

    python el051_build_triggers.py            # write the two files
    python el051_build_triggers.py --check    # detect and report, write nothing

EL051's language mapping (Bern, 2026-09-01) was run in two parts:

    EL051_20260901_09h45m28_15min.h5   09:45  picture block (54 trials), then an audio
                                              block interrupted after 24 trials (a 165 s
                                              break inside it) and abandoned
    EL051_20260901_14h29m54_18min.h5   14:29  the full audio block (53) and the reading
                                              block (53)

and logged in two events tables (the 14h16m6s one is empty, a false start; the 14h16m47s
one starts with two aborted picture rows that have no photodiode pulse and were run
before the recording started).

The pipeline wants one recording and one events table per patient. cfg.RAW_CONCAT makes
the two files one recording (file 2 begins at sample 934016 of the joined axis). This
script reads the photodiode (DC6) of each file separately - the two files have different
DC baselines, so one threshold over the joined trace would not do - pairs the pulses with
the log rows in order, and writes:

    data_LM/prep0/EL051_LM_manual_trigs_concat.tsv
        sample, sample_offsets on the joined axis, 160 rows: 54 picture (file 1), then
        53 audio and 53 reading (file 2). cfg.EL_PRESETS["EL051"]["manual_trig"] points
        at it, so 140's photodiode detection is bypassed.
    data_LM/raw/sub-EL051_task-LanguageMapping_merged_events.tsv
        the matching 160 log rows in the same order (log 1 rows 0-53, log 2 rows 2-107),
        in the Bern format 140 reads for condition / accuracy / exemplar / duration, with
        response_time renamed response_timex as in every other Bern log, so the reader
        takes `duration` for trial_end and not the epoch timestamp. Each log counts
        `onset` from its own start; the log 2 rows are moved into log 1's frame (one
        constant, measured as the pulse-minus-onset offset of each log), because the
        PD-extraction plot draws the log onsets with a single shift taken from the first
        trial - with two frames the audio and reading reference lines sat 8 s early.

What is left out, deliberately: the 51.6 s task-start pulse of file 1 and the 24 morning
audio trials (the block was redone in full in the afternoon). Set KEEP_MORNING_AUDIO to
include them - they then sit between the picture and the afternoon audio trials.

Nothing here touches the original logs or the recordings.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, pandas as pd, h5py

D = r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_BERN\EL051\task_FBM\data_LM"
FILES = ["EL051_20260901_09h45m28_15min.h5", "EL051_20260901_14h29m54_18min.h5"]
LOGS = ["sub-EL051_task-LanguageMapping_timestamp-1-9-2026(9h46m23s)_lang-GER_events.tsv",
        "sub-EL051_task-LanguageMapping_timestamp-1-9-2026(14h16m47s)_lang-GER_events.tsv"]
OUT_TRIGS = os.path.join(D, "prep0", "EL051_LM_manual_trigs_concat.tsv")
OUT_EVENTS = os.path.join(D, "raw", "sub-EL051_task-LanguageMapping_merged_events.tsv")
TRIG = "DC6"
KEEP_MORNING_AUDIO = False


def pulses(path):
    """Photodiode pulses of one file: (onset_sample, offset_sample, fs, n_samples).
    The threshold sits halfway between the low state (30th percentile) and the pulse
    level (99.5th); pulses shorter than 0.2 s are switching noise and dropped."""
    with h5py.File(path, "r") as f:
        g = f["traces/raw"]; fs = float(g.attrs["sfreq"]); x = g[TRIG][()].astype(float)
    xs = np.convolve(x, np.ones(10) / 10, mode="same")
    lo, hi = np.percentile(xs, [30, 99.5]); thr = (lo + hi) / 2
    h = xs > thr
    on = np.flatnonzero(h[1:] & ~h[:-1]) + 1
    off = np.flatnonzero(~h[1:] & h[:-1]) + 1
    if len(off) and len(on) and off[0] < on[0]:
        off = off[1:]
    n = min(len(on), len(off)); on, off = on[:n], off[:n]
    keep = (off - on) / fs > 0.2
    return on[keep], off[keep], fs, len(x)


def read_log(path):
    return pd.read_csv(path, sep="\t", skiprows=1, dtype=str)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="detect and report, write nothing")
    a = ap.parse_args()

    on1, off1, fs, n1 = pulses(os.path.join(D, "raw", FILES[0]))
    on2, off2, fs2, n2 = pulses(os.path.join(D, "raw", FILES[1]))
    assert fs == fs2, (fs, fs2)
    w1, w2 = (off1 - on1) / fs, (off2 - on2) / fs
    L1, L2 = read_log(os.path.join(D, LOGS[0])), read_log(os.path.join(D, LOGS[1]))
    print(f"file 1: {len(on1)} pulses over {n1 / fs:.0f} s; log 1: {len(L1)} rows "
          f"({L1.category.value_counts().to_dict()})")
    print(f"file 2: {len(on2)} pulses over {n2 / fs:.0f} s; log 2: {len(L2)} rows "
          f"({L2.category.value_counts().to_dict()})")

    # ---- file 1: the task-start pulse, 54 picture pulses (1 s each), 24 audio pulses
    assert w1[0] > 30, f"file 1 does not start with the task-start pulse (first width {w1[0]:.1f} s)"
    pic = slice(1, 55); aud1 = slice(55, len(on1))
    assert np.all((w1[pic] > 0.9) & (w1[pic] < 1.1)), "file 1: the 54 pulses after the start pulse are not the 1 s picture pulses"
    n_aud1 = len(on1) - 55
    assert n_aud1 == int((L1.category == "auditory_naming_GER").sum()) == 24, f"file 1: {n_aud1} audio pulses"
    assert int((L1.category == "picture_naming").sum()) == 54
    # ---- file 2: 53 audio pulses (the spoken sentence, 3-8 s) then 53 reading (3.5 s)
    n_aud2 = int((L2.category == "auditory_naming_GER").sum()); n_read = int((L2.category == "reading_completion").sum())
    assert len(on2) == n_aud2 + n_read == 106, f"file 2: {len(on2)} pulses for {n_aud2} + {n_read} log rows"
    assert np.all(np.abs(w2[n_aud2:] - 3.5) < 0.1), "file 2: the last 53 pulses are not the 3.5 s reading pulses"
    l2_pic = (L2.category == "picture_naming").sum()
    assert l2_pic == 2 and (pd.to_numeric(L2.duration[:2]) == 0).all(), "log 2 should start with the two aborted picture rows"

    rows = []   # (sample, sample_offsets, log row)
    for k, i in enumerate(range(pic.start, pic.stop)):
        rows.append((int(on1[i]), int(off1[i]), L1.iloc[k]))
    if KEEP_MORNING_AUDIO:
        for k, i in enumerate(range(aud1.start, aud1.stop)):
            rows.append((int(on1[i]), int(off1[i]), L1.iloc[54 + k]))
    for j in range(len(on2)):
        rows.append((int(on2[j]) + n1, int(off2[j]) + n1, L2.iloc[2 + j]))

    trig = pd.DataFrame({"sample": [r[0] for r in rows], "sample_offsets": [r[1] for r in rows]})
    ev = pd.DataFrame([r[2] for r in rows]).reset_index(drop=True)
    ev = ev.rename(columns={"response_time": "response_timex"})
    # one time frame for `onset`: log 2 rows are shifted so that pulse minus onset is the
    # same constant as in log 1 (each offset is checked to be constant within its log)
    n_log1 = 54 + (n_aud1 if KEEP_MORNING_AUDIO else 0)
    onset = pd.to_numeric(ev["onset"]).to_numpy(); pulse_s = trig["sample"].to_numpy() / fs
    d1, d2 = pulse_s[:n_log1] - onset[:n_log1], pulse_s[n_log1:] - onset[n_log1:]
    assert np.ptp(d1) < 0.1 and np.ptp(d2) < 0.1, f"pulse-minus-onset not constant: log 1 spread {np.ptp(d1):.3f} s, log 2 {np.ptp(d2):.3f} s"
    shift = float(np.median(d2) - np.median(d1))
    ev.loc[n_log1:, "onset"] = [f"{v + shift:.6f}" for v in onset[n_log1:]]
    print(f"log onsets: pulse minus onset is {np.median(d1):.3f} s in log 1 and {np.median(d2):.3f} s in log 2 "
          f"(spread {np.ptp(d1) * 1000:.0f} / {np.ptp(d2) * 1000:.0f} ms); log 2 rows shifted by {shift:+.3f} s into log 1's frame")
    # the log's own sanity: category order and the stimulus widths
    cats = ev.category.tolist()
    print(f"{len(rows)} trials: " + ", ".join(f"{c} {cats.count(c)}" for c in dict.fromkeys(cats)))
    print(f"joined axis: file 2 starts at sample {n1} ({n1 / fs:.1f} s); last trigger at "
          f"{trig.sample_offsets.max() / fs:.1f} s of {(n1 + n2) / fs:.1f} s")
    print("stimulus widths (s): picture %.2f-%.2f, audio %.1f-%.1f, reading %.2f-%.2f" % (
        w1[pic].min(), w1[pic].max(), w2[:n_aud2].min(), w2[:n_aud2].max(), w2[n_aud2:].min(), w2[n_aud2:].max()))
    gaps = np.diff(trig["sample"].values) / fs
    print("gap to the next trial (s): median %.1f, max %.1f (the file boundary sits in it)" % (np.median(gaps), gaps.max()))

    if a.check:
        print("(--check: nothing written)")
        return 0
    os.makedirs(os.path.dirname(OUT_TRIGS), exist_ok=True)
    trig.to_csv(OUT_TRIGS, sep="\t", index=False)
    with open(OUT_EVENTS, "w", encoding="utf-8", newline="") as f:
        f.write("SubjectNumber : EL051\n")
        ev.to_csv(f, sep="\t", index=False, lineterminator="\n")
    print("wrote", OUT_TRIGS); print("wrote", OUT_EVENTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
