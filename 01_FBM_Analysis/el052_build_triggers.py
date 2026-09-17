#!/usr/bin/env python3
"""
el052_build_triggers.py - EL052's trial triggers and one events table, from one recording
and two logs.

    python el052_build_triggers.py            # write the two files
    python el052_build_triggers.py --check    # detect and report, write nothing

EL052's language mapping (Bern, 2026-09-15) ran in one 27-minute recording
(EL052_20260915_11h54m53_27min_HUG_LM.h5, DC6 = photodiode) but the blocks ran in the
order AUDIO, READING, PICTURE, and were logged in two files:

    ...11h50m33s..._events.tsv   two picture trials started before the recording and
                                 abandoned (no pulse in the file), then the audio block
                                 (53) and the reading block (53)
    ...12h17m10s..._events.tsv   the picture block (54), then one audio trial started at
                                 the very end and abandoned

Three things about this photodiode differ from EL051's and matter for notebook 140:
the trace is INVERTED (the screen is bright between trials and dark while a stimulus is
on, so the stimulus is a dip - flip=True), the file is Blosc-compressed (h5py needs
`import hdf5plugin` or it cannot read a single sample), and two short dips sit at the
very end of the file (1644 s and 1648 s, after the last picture) that are not trials.

This script detects the dips itself (50 ms box smoothing against the screen flicker,
threshold midway between the 30th and 99.5th percentile of the flipped trace, dips
shorter than 0.2 s dropped), checks them against the logs block by block - pulse gaps
against log gaps to a fraction of a second, 53 / 53 / 54 - and writes:

    data_LM/prep0/EL052_LM_manual_trigs.tsv
        sample, sample_offsets - 160 rows in recording order (53 audio, 53 reading,
        54 picture). cfg.EL_PRESETS["EL052"]["manual_trig"] points at it, so 140's
        detection is bypassed and the two trailing dips never enter.
    data_LM/raw/sub-EL052_task-LanguageMapping_merged_events.tsv
        the matching 160 log rows in the same order, Bern format, response_time renamed
        response_timex (so the reader takes `duration` for trial_end, not the epoch),
        with log 2's onsets moved into log 1's time frame by the constant the pulses
        give, so the PD-extraction plot's single-shift reference lines land on all
        three blocks.

Left out, deliberately: the two aborted picture rows and the aborted audio row. The two
original logs are not touched - but they must not stay in data_LM/raw beside the merged
table, because 140 takes the first *.tsv it finds there.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, pandas as pd
try:
    import hdf5plugin  # noqa: F401  - registers the Blosc filter the file needs
except ImportError:
    pass
import h5py

D = r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_BERN\EL052\task_FBM\data_LM"
H5 = "EL052_20260915_11h54m53_27min_HUG_LM.h5"
LOGS = ["sub_EL052_task_LanguageMapping_timestamp_15_9_2026_11h50m33s_lang_GER_events.tsv",
        "sub_EL052_task_LanguageMapping_timestamp_15_9_2026_12h17m10s_lang_GER_events.tsv"]
OUT_TRIGS = os.path.join(D, "prep0", "EL052_LM_manual_trigs.tsv")
OUT_EVENTS = os.path.join(D, "raw", "sub-EL052_task-LanguageMapping_merged_events.tsv")
TRIG = "DC6"
END_S = 1640.0          # nothing after this is a trial: two stray dips at 1644 and 1648 s


def dips(path):
    """Photodiode dips of the flipped trace: (onset_sample, offset_sample, fs, n_samples)."""
    with h5py.File(path, "r") as f:
        g = f["traces/raw"]; fs = float(g.attrs["sfreq"]); x = -g[TRIG][()].astype(float)
    k = int(0.05 * fs)
    xs = np.convolve(x, np.ones(k) / k, mode="same")
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

    on, off, fs, n = dips(os.path.join(D, "raw", H5))
    w = (off - on) / fs; t = on / fs
    L1, L2 = read_log(os.path.join(D, "raw", LOGS[0])), read_log(os.path.join(D, "raw", LOGS[1]))
    print(f"recording: {n / fs:.0f} s at {fs:.0f} Hz; {len(on)} dips > 0.2 s on {TRIG} (flipped)")
    print(f"log 1: {len(L1)} rows ({L1.category.value_counts().to_dict()}); log 2: {len(L2)} rows ({L2.category.value_counts().to_dict()})")

    # the logs, in recording order: log 1's audio and reading, then log 2's pictures
    aud = L1[L1.category == "auditory_naming_GER"]; rea = L1[L1.category == "reading_completion"]
    pic = L2[L2.category == "picture_naming"]
    assert len(aud) == 53 and len(rea) == 53 and len(pic) == 54, (len(aud), len(rea), len(pic))
    assert (L1.category.iloc[:2] == "picture_naming").all() and L2.category.iloc[-1] == "auditory_naming_GER", "unexpected aborted rows"
    ev = pd.concat([aud, rea, pic]).reset_index(drop=True)

    # the dips: everything before END_S, in order; the trailing dips are not trials
    late = t >= END_S
    print(f"dips after {END_S:.0f} s, dropped: {np.round(t[late], 1).tolist()}")
    on, off, w, t = on[~late], off[~late], w[~late], t[~late]
    assert len(on) == len(ev) == 160, f"{len(on)} dips for {len(ev)} log rows"
    assert t[0] > 100, f"a dip before the audio block, at {t[0]:.1f} s"

    # block by block: widths, and pulse gaps against log gaps
    on_s = pd.to_numeric(ev.onset).to_numpy()
    for name, sl, wlo, whi in (("audio", slice(0, 53), 3.0, 8.0), ("reading", slice(53, 106), 3.4, 3.6), ("picture", slice(106, 160), 0.9, 1.1)):
        ww = w[sl]; gap_p = np.diff(t[sl]); gap_l = np.diff(on_s[sl])
        assert np.all((ww > wlo) & (ww < whi)), f"{name}: dip widths {ww.min():.2f}-{ww.max():.2f} s outside {wlo}-{whi}"
        assert np.abs(gap_p - gap_l).max() < 0.5, f"{name}: pulse gaps and log gaps disagree by up to {np.abs(gap_p - gap_l).max():.2f} s"
        print(f"  {name:8s} {sl.stop - sl.start} dips, widths {ww.min():.2f}-{ww.max():.2f} s, gaps agree with the log to {np.abs(gap_p - gap_l).max() * 1000:.0f} ms")

    # one time frame for `onset`: log 2 rows are shifted so that pulse minus onset is the same constant as in log 1
    d1 = t[:106] - on_s[:106]; d2 = t[106:] - on_s[106:]
    assert np.ptp(d1) < 0.1 and np.ptp(d2) < 0.1, f"pulse-minus-onset not constant: log 1 spread {np.ptp(d1):.3f} s, log 2 {np.ptp(d2):.3f} s"
    shift = float(np.median(d2) - np.median(d1))
    ev = ev.rename(columns={"response_time": "response_timex"})
    ev.loc[106:, "onset"] = [f"{v + shift:.6f}" for v in on_s[106:]]
    print(f"log onsets: pulse minus onset is {np.median(d1):.3f} s in log 1 and {np.median(d2):.3f} s in log 2; log 2 rows shifted by {shift:+.3f} s into log 1's frame")
    trig = pd.DataFrame({"sample": on.astype(int), "sample_offsets": off.astype(int)})
    print(f"160 trials: audio 53, reading 53, picture 54 | first at {t[0]:.1f} s, last at {t[-1]:.1f} s of {n / fs:.1f}")

    if a.check:
        print("(--check: nothing written)")
        return 0
    os.makedirs(os.path.dirname(OUT_TRIGS), exist_ok=True)
    trig.to_csv(OUT_TRIGS, sep="\t", index=False)
    with open(OUT_EVENTS, "w", encoding="utf-8", newline="") as f:
        f.write("SubjectNumber : EL052\n")
        ev.to_csv(f, sep="\t", index=False, lineterminator="\n")
    print("wrote", OUT_TRIGS); print("wrote", OUT_EVENTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
