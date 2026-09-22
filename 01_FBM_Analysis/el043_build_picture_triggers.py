#!/usr/bin/env python3
"""
el043_build_picture_triggers.py - EL043's picture-naming triggers from the 18 June recording,
written on the axis of the two recordings joined (cfg.RAW_CONCAT["EL043"]).

    python el043_build_picture_triggers.py --check    # detect and report, write nothing
    python el043_build_picture_triggers.py            # write the table, park the old one

WHY. EL043's language mapping of 17 June 2025 (the 63-min file) had a bad picture-naming
block; picture naming was redone on 18 June in a second recording,
raw/2ndrun_PictureNaming (877 s, the same 140 channels at 1024 Hz). Until 2026-09-21 the
prep0 picture table was the BAD 17 June block (its onsets run to 975 s, longer than the
18 June file), and cfg.EL_PRESETS["EL043"] cropped it away (time_range from 1810 s), so the
fmax-400 run of 2026-09-18 died on picture ("block_window: no onsets") before reading ran.

WHAT. The pipeline wants one recording per patient, so cfg.RAW_CONCAT now joins the two
files: file 2 begins at sample 3864064 (3773.5 s) of the joined axis, and the preset's
window runs 1810..4650 s. The audio and reading tables in prep0 are file 1's and are
already on that axis. This script makes the picture table:

    1. the 18 June file alone, DC6 flipped, window 31..863 s - the detector and the log
       pairing 140's photodiode cell uses (LFfunctions_PDextract.get_trigger_indexes_photodiode,
       threshold 0.40), against the 18 June events log (54 picture rows);
    2. every sample index shifted by file 1's length;
    3. the old picture table moved to prep0/prep0-bad/ (the convention PAT_6953 uses),
       and the new one written by LFfunctions_PDextract.parse_and_save, i.e. in exactly
       the format of the other prep0 tables:
           prep0/EL043_LM_picture__DC6_<today>.tsv

Then run 140 for EL043 once; it reads prep0 and the joined recording and produces all
three conditions.

Nothing here touches the recordings or the logs.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import tempfile
from datetime import date

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import LFfunctions_PDextract as LF                  # noqa: E402
from functions import config as cfg                 # noqa: E402

D = r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_BERN\EL043\task_FBM\data_LM"
FILE1 = "EL043_20250617_16h01m36_63min_HUG_LM.h5"
FILE2 = os.path.join("2ndrun_PictureNaming", "EL043_20250618_11h04m06_15min_HUG_LM.h5")
LOG2 = os.path.join("2ndrun_PictureNaming",
                    "sub_EL043_task_LanguageMapping_timestamp_18_6_2025_11h3m33s_lang_GER_events.tsv")
# the 18 June file on its own - the commented line under cfg.EL_PRESETS["EL043"]
PRESET = dict(trig="DC6", flip=True, time_range=(31, 863), invalid_trials=[], fake_trials=[],
              trial_ids=["picture"] * 54)
THRESHOLD = 0.40          # what 140's photodiode cell passes


def load_h5(path, only=None):
    """(samples x channels), fs, names - the traces/raw schema, as lf_io_utils.load_h5 reads it.
    `only` = the channels to load (the detector needs the photodiode alone); the returned
    `all_names` is the full channel list, for the RAW_CONCAT check."""
    with h5py.File(path, "r") as f:
        g = f["traces/raw"]
        all_names = list(g.keys())
        names = [c for c in all_names if c in set(only)] if only else all_names
        fs = float(g.attrs.get("sfreq", f.attrs.get("sfreq", np.nan)))
        X = np.stack([g[ch][()] for ch in names], axis=1)
    return X, fs, names, all_names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="detect and report, write nothing")
    a = ap.parse_args()
    raw, prep0 = os.path.join(D, "raw"), os.path.join(D, "prep0")

    with h5py.File(os.path.join(raw, FILE1), "r") as f:
        g = f["traces/raw"]; names1 = list(g.keys()); fs1 = float(g.attrs["sfreq"]); n1 = int(g[names1[0]].shape[0])
    X2, fs, names2, all2 = load_h5(os.path.join(raw, FILE2), only=[PRESET["trig"]])
    assert fs == fs1, f"sampling rates differ: {fs1} vs {fs}"
    assert names1 == all2, "the two files do not carry the same channels - RAW_CONCAT would refuse them"
    assert cfg.RAW_CONCAT.get("EL043", [None])[0] == FILE1, "cfg.RAW_CONCAT['EL043'] does not start with the 17 June file"
    print(f"file 1: {n1} samples = {n1 / fs:.1f} s at {fs:g} Hz, {len(names1)} channels")
    print(f"file 2: {X2.shape[0]} samples = {X2.shape[0] / fs:.1f} s, {len(all2)} channels; joined axis puts it at {n1} = {n1 / fs:.1f} s")

    # The 18 June log writes the answer's epoch timestamp in a column named `response_time`,
    # and the log reader takes THAT as the trial duration when it exists (it only falls back
    # to `duration` otherwise) - which is why the 17 June log carries `response_timex`, as
    # every other Bern log does. The detector and the writer get a copy with the same rename.
    log2_src = os.path.join(raw, LOG2)
    log2 = (os.path.join(tempfile.gettempdir(), "EL043_log2_response_timex.tsv") if a.check
            else log2_src.replace("_events.tsv", "_events_response_timex.tsv"))
    with open(log2_src, encoding="utf-8") as f:
        lines = f.read().splitlines()
    lines[1] = "\t".join("response_timex" if c == "response_time" else c for c in lines[1].split("\t"))
    with open(log2, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    L2 = pd.read_csv(log2, sep="\t", skiprows=1, dtype=str)
    assert int((L2.category == "picture_naming").sum()) == 54 == len(L2), f"log 2: {L2.category.value_counts().to_dict()}"
    assert "response_timex" in L2.columns and "response_time" not in L2.columns

    # do_plot=True: the trial ends (log duration after each pulse) are only computed by the
    # plotting routine; the figure is kept as the check figure
    on, off, metrics = LF.get_trigger_indexes_photodiode(
        raw_signals=X2, sampling_rate=fs, channel_names=names2, trig_name=PRESET["trig"],
        time_range=PRESET["time_range"], threshold_val=THRESHOLD, flip_trigs=PRESET["flip"],
        do_plot=True, trial_ids=PRESET["trial_ids"], invalid_trials=PRESET["invalid_trials"],
        ignore_invalid=False, fake_trials=PRESET["fake_trials"], extra_table_path=log2,
        extra_verbose=False, return_extra_metrics=True,
    )
    fig_dir = os.path.join(HERE, "outputs", "04_ersp_LM", "EL043", "LM", "Report")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "EL043_picture_18June_PDcheck.png")
    plt.gcf().savefig(fig_path, dpi=150); plt.close("all")
    print(f"check figure: {fig_path}")
    on, off = np.asarray(on, dtype=np.int64), np.asarray(off, dtype=np.int64)
    w = (off - on) / fs
    print(f"detector: {len(on)} pulses on {PRESET['trig']} (flipped) in {PRESET['time_range']} s; "
          f"widths {w.min():.2f}-{w.max():.2f} s; onsets {on[0] / fs:.1f}..{on[-1] / fs:.1f} s of file 2")
    assert len(on) == 54, f"expected the 54 picture pulses, got {len(on)}"
    assert np.all((w > 0.9) & (w < 1.1)), "not all pulses are the 1 s picture pulses"

    # onto the joined axis
    on_j, off_j = on + n1, off + n1
    if isinstance(metrics, dict):
        for k in ("duration_end_abs", "anchor_abs"):
            v = metrics.get(k)
            if v is not None:
                metrics[k] = np.asarray(v, dtype=np.int64) + n1
    te = metrics.get("duration_end_abs") if isinstance(metrics, dict) else None
    assert te is not None and len(te) == 54, "no trial ends came back from the log pairing"
    print(f"trial ends: log duration after each pulse, {np.min((te - off_j) / fs):.1f}-{np.max((te - off_j) / fs):.1f} s after the offsets")
    print(f"joined axis: first picture onset {on_j[0] / fs:.1f} s, last {on_j[-1] / fs:.1f} s"
          + (f", last trial end {np.max(te) / fs:.1f} s" if te is not None else "")
          + f" - inside the preset window {cfg.EL_PRESETS['EL043']['time_range']}")
    t0, t1 = cfg.EL_PRESETS["EL043"]["time_range"]
    assert on_j[0] / fs > t0 and (np.max(te) if te is not None else off_j[-1]) / fs < t1, "the picture trials do not fit the preset window"

    old = sorted(glob.glob(os.path.join(prep0, "EL043_LM_picture__*.tsv")))
    if a.check:
        print(f"--check: nothing written. Would park {[os.path.basename(x) for x in old]} in prep0/prep0-bad/ and write "
              f"prep0/EL043_LM_picture__DC6_{date.today().isoformat()}.tsv")
        return 0

    bad_dir = os.path.join(prep0, "prep0-bad")
    os.makedirs(bad_dir, exist_ok=True)
    for x in old:
        shutil.move(x, os.path.join(bad_dir, os.path.basename(x)))
        print(f"parked the bad 17 June table: prep0-bad/{os.path.basename(x)}")
    LF.parse_and_save("EL043", "EL043", on_j, off_j, metrics, fs, D, "LM", log2,
                      PRESET["trial_ids"], PRESET["trig"], cond_alias=getattr(cfg, "COND_ALIAS", None))
    new = sorted(glob.glob(os.path.join(prep0, "EL043_LM_picture__*.tsv")))
    assert len(new) == 1, f"expected one picture table in prep0, found {new}"
    t = pd.read_csv(new[0], sep="\t")
    assert len(t) == 54 and int(t["sample"].min()) > n1, "the written table is not the shifted 54-trial one"
    print(f"wrote {os.path.basename(new[0])}: {len(t)} trials, onsets {t['onset'].min():.1f}..{t['onset'].max():.1f} s, "
          f"samples {int(t['sample'].min())}..{int(t['sample'].max())}, resp_accuracy {t['resp_accuracy'].value_counts().to_dict()}")
    print("next: python 140_ersp_pipeline.py --patient EL043")
    return 0


if __name__ == "__main__":
    sys.exit(main())
