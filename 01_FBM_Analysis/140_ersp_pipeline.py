#!/usr/bin/env python
"""140_ERSP_analysis_pipeline.ipynb as a terminal script - one patient per call.

    python .\140_ersp_pipeline.py --patient EL030
    python .\140_ersp_pipeline.py --patient PAT_6684
    python .\140_ersp_pipeline.py --patient G-01            (MicroEPI .mat patient -> PAT_5515)
    python .\140_ersp_pipeline.py --patient EL030 --pd      (also redo the photodiode trial tables first)
    python .\140_ersp_pipeline.py --patient EL030 --no-qc   (cubes and CLEAN PNGs only, no ERSP / HFA / PSD figures)

Run it from anywhere; it changes into 01_FBM_Analysis itself, the way the notebook kernel
sits there. Use the same Python the notebook runs in (it needs mne, h5py, scipy).

WHY A SCRIPT. The notebook holds every patient's intermediates in one kernel and dies
partway through a 32-patient run. Here each patient is one process: it starts clean,
writes its outputs, prints the same lines the notebook printed, appends its row to
outputs/03_ERSP/wm_reref_report.tsv, and exits. A crash costs one patient.

WHAT IT IS. Cells 1, 2, the PD-extraction cell and the pipeline cell of the notebook,
verbatim in their logic (2026-09-18), with three differences:
  - the patient comes from --patient, not from cfg.patient_ids or the per-cell overrides;
  - photodiode extraction runs only with --pd (the prep0 tables already exist);
  - the report TSV is merged per patient rather than rewritten per batch.
The output trees and everything in them are the ones the notebook writes.

Everything is logged to outputs/03_ERSP/logs/<patient>_<timestamp>.log as well as to
the terminal.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np                          # noqa: E402
import pandas as pd                         # noqa: E402
import matplotlib                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt             # noqa: E402

from functions import lf_io_utils as io, lf_trials as tr, lf_ersp as fe, config as cfg   # noqa: E402
from functions.config import (PAT_PRESETS, EL_PRESETS, MICROEPI_MAT_PRESETS, COND_ALIAS)  # noqa: E402
import LFfunctions_PDextract as LF          # noqa: E402
from LF_pd import load_patient_raw          # noqa: E402
import functions.lf_micromacro as mm        # noqa: E402


class _Tee:
    """stdout to the terminal and to a log file, line by line."""
    def __init__(self, path):
        self._f = open(path, "a", encoding="utf-8")
        self._o = sys.stdout

    def write(self, s):
        self._o.write(s)
        self._f.write(s)

    def flush(self):
        self._o.flush()
        self._f.flush()


# ============================================================
# Cell 1 - run controls (the toggles become flags)
# ============================================================
BLOCK = "LM"
# ONE TREE since 2026-09-30 (cfg.ERSP_TREE = 03_ERSP): the QC figures and the cubes of a
# patient land in the same <pid>/LM/ folder. The two names are kept because every writer
# below picks one of them; they simply point at the same place now. 04_ersp_LM and
# 04_ersp_LM_RAWONLY are frozen as the previous way.
ERSP_SCRIPT_NAME    = cfg.ERSP_TREE
RAWONLY_SCRIPT_NAME = cfg.ERSP_TREE
run_root_ersp       = os.path.join(cfg.outputs_root, ERSP_SCRIPT_NAME)
run_root_raw        = os.path.join(cfg.outputs_root, RAWONLY_SCRIPT_NAME)

# PRODUCT SWITCHES. Every one defaults to True, so a normal run writes exactly what it
# wrote before these existed. They are here for 145_trial_zscore_test.py, which needs the
# same pipeline but only two of the products; without them that script would have to fork
# this loop, and a forked 873-line pipeline drifts from the real one within a week.
WRITE_MONTAGE      = True   # Report/<pid>_montage_overview.png
WRITE_ERSP_PLOTS   = True   # ERSP/<cond>/*.png          (the per-channel ERSP figures)
WRITE_HG_PLOTS     = True   # HFA/<cond>/*.png           (the per-trial high-frequency rasters; "HG" until 09-30)
WRITE_SIGNAL_PLOTS = True   # Signal/<cond>/*.png        (the same rows as traces of the cleaned signal, 2026-10-02)
WRITE_CUBES        = True   # ERSP_matrix/<cond>/*.npy   (what stage 02 clusters)
WRITE_HALVES       = True   # ERSP_halves/<cond>/*.npy   (the split-half gate)
WRITE_CLEAN_PNG    = True   # ERSP_clean/<cond>/*_CLEAN.png
EXPORT_TRIAL_SCORES = True   # TrialScores/<cond>/<pid>_<cond>_trial_scores.tsv (in the 03_ERSP tree since 09-30; was 145's only)

ersp_params = fe.ERSPParams(
    nperseg=cfg.nperseg, nfft=cfg.nfft, noverlap=cfg.noverlap,
    baseline_w=cfg.baseline_w, baseline_calc_w=cfg.baseline_calc_w,   # (-0.4, -0.1): the ERSPParams default every cube was made with; explicit since 2026-09-24
    proportions=cfg.proportions,
    n_time_bins=cfg.n_time_bins, vmin=cfg.vmin, vmax=cfg.vmax, fmax=cfg.fmax
)

# ============================================================
# Cell 2 - helpers (implementations live in lf_ersp.py and lf_io_utils.py)
# ============================================================
notch_mains_harmonics = fe.notch_mains_harmonics
fill_nans_nearest     = fe.fill_nans_nearest
save_clean_png        = fe.save_clean_png
plot_psd_overview     = fe.plot_psd_overview
_is_non_neural        = io._is_non_neural


def _is_photodiode(name) -> bool:
    """The HUG photodiode channel ("photo" / "PHOTO"): kept for its figures, never data."""
    return str(name).strip().upper().startswith("PHOTO")


_ensure               = io.ensure_dir


def notch_method_for(patient_id, pid_raw):
    """cfg.notch_method: {patient: "iir" | "interp"}; cfg.notch_method_default when unlisted.

    The default was "iir" written in here until 2026-09-24, so a patient nobody had
    thought about got the method that digs holes. It is cfg's call now, and cfg says
    interp for everyone.
    """
    dflt = str(getattr(cfg, "notch_method_default", "iir"))
    m = getattr(cfg, "notch_method", {})
    return str(m.get(str(patient_id), m.get(str(pid_raw), dflt)))


def notch_q_max_for(patient_id, pid_raw):
    """cfg.notch_Q_max: {patient: Q}; 500 (the formula's own ceiling) when unlisted."""
    q = getattr(cfg, "notch_Q_max", {})
    return float(q.get(str(patient_id), q.get(str(pid_raw), 500.0)))


def notch_per_shaft_for(patient_id, pid_raw):
    """cfg.notch_shaft_all, else cfg.notch_shaft_patients: IDs or substrings."""
    if bool(getattr(cfg, "notch_shaft_all", False)):
        return True
    pats = getattr(cfg, "notch_shaft_patients", [])
    return any(s in str(pid_raw) or s in str(patient_id) for s in pats)


def apply_notch_with_audit(signals, fs, patient_id, pid_raw, audit=None, names=None):
    return fe.apply_notch_with_audit(
        signals, fs, patient_id, pid_raw,
        notch_patients=getattr(cfg, "notch_patients", []),
        mains_base=getattr(cfg, "mains_base", 50.0),
        fmax=getattr(cfg, "fmax", 500.0),
        repeats=getattr(cfg, "notch_repeats", 1),
        peak_z_thresh=getattr(cfg, "notch_peak_z_thresh", 3.0),
        extra_bases=tuple(getattr(cfg, "notch_extra_bases", {}).get(patient_id, ())),
        audit=audit,
        per_shaft=bool(names is not None and notch_per_shaft_for(patient_id, pid_raw)),
        names=names,
        Q_max=notch_q_max_for(patient_id, pid_raw),
        method=notch_method_for(patient_id, pid_raw),
        interp_kw=dict(phase=getattr(cfg, "notch_interp_phase", "random"),
                       max_hw_hz=float(getattr(cfg, "notch_interp_max_hw_hz", 12.0)),
                       min_hw_hz=dict(getattr(cfg, "notch_interp_min_hw_hz", {}).get(patient_id, {}))),
    )


def apply_wm_reref(signals, names, patient_id, *, electrodes_tsv_pattern=None):
    """Returns (signals, reref_label, wm_skip_set). Raises ValueError if cfg.reref_type is
    not 'WM', if no WM channels are found, or if WM rereferencing failed to apply."""
    if str(cfg.reref_type).upper() != "WM":
        raise ValueError(
            f"[reref] cfg.reref_type is '{cfg.reref_type}' — only 'WM' is "
            f"allowed in this pipeline. Update config.py and re-run.")
    wm_idx = io.wm_indices_for_patient(patient_id, names,
                                        electrodes_tsv_pattern=electrodes_tsv_pattern)
    if not wm_idx:
        raise ValueError(
            f"[reref] {patient_id}: no WM channels found — cannot apply WM "
            f"rereferencing. Check the electrodes TSV and WM threshold.")
    bad = getattr(cfg, "bad_channels_manual", {}).get(patient_id, [])
    signals_r, used, excluded = fe.apply_wm_reference_with_exclusions(
        signals, names, wm_idx, bad)
    if not used:
        raise ValueError(
            f"[reref] {patient_id}: WM channels were found but none were used "
            f"after exclusions — cannot guarantee WM rereferencing. "
            f"Check bad_channels_manual and available WM channels.")
    return signals_r, "WM", set(used) | set(excluded)


def _load_signals_and_prep_for_patient(pid_raw):
    """Route a patient to its loader (MicroEPI .mat via lf_micromacro; PAT/EL via the
    TRC/EDF/H5 loader). Returns patient_id, signals, names, fs, prep_dir,
    electrodes_tsv_pattern, is_micro."""
    pid_str = str(pid_raw)
    if pid_str in getattr(cfg, "MICROEPI_MAT_PATIENTS", []):
        preset = cfg.MICROEPI_MAT_PRESETS[pid_str]
        patient_id = preset["pat_name"]
        d = mm.load_and_concatenate_mats(preset["data_dir"], preset["mat_files"])
        signals, names, is_micro = mm.build_combined_signals(
            d["data_ecog"], d["data_micro"], d["chans_ecog"], d["chans_micro"])
        fs = d["fs"]
        prep_dir = os.path.join(os.path.dirname(preset["data_dir"]), "prep0")
        electrodes_tsv = preset["electrodes_tsv"]
        print(f"  [paths] data_dir: {preset['data_dir']}")
        print(f"  [paths] prep_dir: {prep_dir}")
        print(f"  [microepi] {signals.shape[1]} channels = "
              f"{int((~is_micro).sum())} macros + {int(is_micro.sum())} micros")
    else:
        patient_id, raw_dir, prep_dir = io.build_paths_for_patient(pid_raw, cfg.block_name)
        signals, names, fs = io.load_raw_for_patient(patient_id, raw_dir)   # cfg.RAW_CONCAT joins a split recording
        electrodes_tsv = None  # auto-resolve from cohort
        is_micro = None
    return patient_id, signals, names, fs, prep_dir, electrodes_tsv, is_micro


def out_id_for(pid_raw):
    """The output folder name a raw id maps to (PAT_5515 for G-01, PAT_3455 for 3455)."""
    s = str(pid_raw)
    if s in getattr(cfg, "MICROEPI_MAT_PATIENTS", []):
        return cfg.MICROEPI_MAT_PRESETS[s]["pat_name"]
    if s.startswith("EL") or s.startswith("PAT_"):
        return s
    return f"PAT_{s}"


# ============================================================
# Part 1 - photodiode / trial extraction (--pd), the notebook's loop for ONE patient
# ============================================================
def run_pd_extraction(pid_raw):
    s = str(pid_raw)
    if s in getattr(cfg, "MICROEPI_MAT_PATIENTS", []):
        preset = cfg.MICROEPI_MAT_PRESETS.get(s)
        if preset is None:
            print(f"\n[skip] MicroEPI-{s}: no preset in cfg.MICROEPI_MAT_PRESETS"); return
        patient_id = preset["pat_name"]
        print(f"\n=== MicroEPI-{s} → {patient_id} ===")
        try:
            d = mm.load_and_concatenate_mats(preset["data_dir"], preset["mat_files"])
            fs = d["fs"]
            print(f"  signals: ecog {d['data_ecog'].shape}  micro {d['data_micro'].shape}  fs={fs}")
            beh_path = os.path.join(preset["data_dir"], preset["tsv_file"])
            on_abs, off_abs = mm.extract_events_from_photodiode(
                d["photodiode"], fs,
                time_range=preset.get("time_range", (0, -1)),
                trial_ids=preset.get("trial_ids", []) or None,
                invalid_trials=preset.get("invalid_trials", []) or None,
                fake_trials=preset.get("fake_trials", []) or None,
                flip_trigs=True, extra_table_path=beh_path, do_plot=False,
            )
            print(f"  photodiode: {len(on_abs)} onsets, {len(off_abs)} offsets")
            beh = LF._read_trial_table(beh_path)
            dfl = beh["raw_df"].rename(columns=str.lower)

            def _pick(cols):
                return next((dfl[c].astype(str).to_numpy() for c in cols if c in dfl), None)
            condition_name = _pick(["category", "blockname"])
            resp_accuracy  = _pick(["response_type", "responseaccuracy"])
            trial_idx_col  = _pick(["exemplar", "stimnumber"])
            trial_ids_for_save = (preset.get("trial_ids") or
                                  ([str(x).lower() for x in condition_name]
                                   if condition_name is not None else []))
            prep_dir = os.path.join(os.path.dirname(preset["data_dir"]), "prep0")
            LF.save_onsets_offsets_by_condition(
                patient_id=patient_id, block_name=BLOCK,
                onsets=on_abs, offsets=off_abs, sampling_rate=fs,
                trial_ids=trial_ids_for_save, out_dir=prep_dir,
                condition_name=condition_name, resp_accuracy=resp_accuracy,
                trial_idx=trial_idx_col, cond_alias=COND_ALIAS,
                trigger_label=preset.get("trig", "photodiode"),
            )
            print(f"  [{patient_id}] PD extraction done -> {prep_dir}")
        except Exception as e:
            print(f"[error] MicroEPI {s}: {e}")
        return

    is_el = s.startswith("EL")
    group = "EL" if is_el else "PAT"
    presets = EL_PRESETS if is_el else PAT_PRESETS
    key = s if is_el else (s if s in presets else (int(s[4:]) if s.startswith("PAT_") and s[4:].isdigit() else s))
    preset = presets.get(key) if key in presets else presets.get(out_id_for(s))
    if preset is None:
        print(f"[skip] {s}: no preset"); return
    patient_id = out_id_for(s)
    print(f"\n=== {patient_id} ===")
    try:
        if group == "PAT":
            _, base_path, _ = io.build_paths_for_patient(key, BLOCK)
            # load_raw_for_patient, not the first .TRC in the folder: a patient listed in
            # cfg.RAW_CONCAT was recorded in several files and its task can sit anywhere
            # across them (PAT_1327's starts in the last third of file 1 and ends in file
            # 2). Taking [0] gave the detection a recording that stops mid-task, with
            # nothing in the output to say so.
            raw_signals, channel_names, sampling_rate = io.load_raw_for_patient(out_id_for(s), base_path)
            save_path = os.path.join(os.path.dirname(base_path), "prep0")
            exp_files = glob.glob(os.path.join(base_path, "*.tsv")) or glob.glob(os.path.join(base_path, "*.txt"))
            exp_file  = exp_files[0] if exp_files else None
        else:
            info          = load_patient_raw(s, block_name=BLOCK, use_el_mat_fallback=False, verbose=True)
            raw_signals   = info["raw_signals"]
            sampling_rate = info["sampling_rate"]
            channel_names = list(info["channel_names"])
            print(channel_names)
            save_path     = info["save_path"]
            exp_file      = info["matching_files_onsets"][0] if info["matching_files_onsets"] else None
        lower_map = {str(c).lower(): str(c) for c in channel_names}
        trig_key  = preset["trig"].lower()
        if trig_key not in lower_map:
            print(f"  [warn] trigger '{preset['trig']}' not found — skipping"); return
        pd_name = lower_map[trig_key]
        mt = preset["manual_trig"]
        manual_path = None
        if mt:
            manual_path = mt if os.path.isabs(mt) else os.path.join(save_path, mt)
            if not os.path.exists(manual_path):
                print(f"  [warn] manual triggers not found: {manual_path}")
                manual_path = None
        print("FLIPS!!! ", preset["flip"])
        on_abs, off_abs, metrics = LF.get_trigger_indexes_photodiode(
            raw_signals=raw_signals, sampling_rate=sampling_rate,
            channel_names=channel_names, trig_name=pd_name,
            time_range=preset["time_range"], threshold_val=0.40,
            flip_trigs=preset["flip"],
            trial_ids=preset["trial_ids"], invalid_trials=preset["invalid_trials"],
            ignore_invalid=False, fake_trials=preset["fake_trials"],
            extra_table_path=exp_file, manual_trigs_path=manual_path,
            return_extra_metrics=True,
        )
        print(f"  Paired trials: {len(on_abs)}")
        LF.parse_and_save(key, patient_id, on_abs, off_abs, metrics,
                          sampling_rate, save_path, BLOCK, exp_file,
                          preset["trial_ids"], preset["trig"], cond_alias=COND_ALIAS)
    except Exception as e:
        print(f"[error] {patient_id}: {e}")


# ============================================================
# Part 2 - ERSP pipeline + cluster export, the notebook's process_patient() verbatim
# ============================================================
def _hg_trial_z(res, hg_all):
    """Per-trial z of the high-gamma score, laid out in the HG figure's row order.

    compute_ersp scores only the trials it was given; the HG figure draws the WHOLE
    trial table, including the ones a filter removed upstream. hg_all["keep_pos"] is the
    map between the two, so the score is written to the row it belongs to and every
    other row stays NaN - which the figure draws as no label rather than as a zero.

    The z is computed within the channel, across its trials, which is the definition
    146_trial_zscore_sweep.py uses. Two definitions of "the z of a trial" would be one
    too many: the number on the figure has to be the number the rule would act on.
    """
    sc = res.get("trial_scores_hg")
    if not sc:
        return None
    sc = np.asarray(sc, float)
    ok = np.isfinite(sc)
    if ok.sum() < 4:
        return None
    sd = float(np.std(sc[ok], ddof=1))
    if sd <= 1e-9:
        return None
    z = (sc - float(np.mean(sc[ok]))) / sd
    if hg_all is None:
        return z
    out = np.full(len(hg_all["on"]), np.nan)
    pos = np.asarray(hg_all["keep_pos"], int)
    n = min(len(pos), len(z))
    out[pos[:n]] = z[:n]
    return out


def process_patient(pid_raw, RUN_ERSP_PIPELINE, RUN_CLUSTER_EXPORT, DO_MONTAGE_PSD_PLOTS,
                    INCLUDE_MICROEPI_MICROS):
    # the patient's own entry first, then the "default" key (2026-09-30); a table with
    # neither leaves that rule off for the patient
    def _rej(name):
        d = getattr(cfg, name, {}) or {}
        return d.get(str(pid_raw), d.get("default"))
    ersp_params.trial_reject_mad = _rej("ersp_trial_reject")
    ersp_params.trial_reject_z   = _rej("ersp_trial_reject_z")
    ersp_params.trial_reject_hg_mad = _rej("ersp_trial_reject_hg_mad")
    ersp_params.trial_reject_hg_z   = _rej("ersp_trial_reject_hg_z")
    _rej_on = (ersp_params.trial_reject_mad or ersp_params.trial_reject_z
               or ersp_params.trial_reject_hg_mad or ersp_params.trial_reject_hg_z)
    if _rej_on:
        print(f"  [{pid_raw}] ERSP trial rejection ON  "
              f"MAD k={ersp_params.trial_reject_mad}  z k={ersp_params.trial_reject_z}  "
              f"HG MAD k={ersp_params.trial_reject_hg_mad}  HG z k={ersp_params.trial_reject_hg_z}")

    report = {
        "pid_raw": str(pid_raw), "patient_id": "", "status": "",
        "n_channels_in": 0, "n_channels_neural": 0, "n_channels_unknown_dropped": 0,
        "n_channels_used": 0, "n_wm_used": 0, "wm_channels_used": "",
        "wm_channels_excluded_as_bad": "", "error": "",
    }
    try:
        patient_id, signals, names, fs, prep_dir, _wm_tsv, is_micro = \
            _load_signals_and_prep_for_patient(pid_raw)
        report["patient_id"] = patient_id
        report["n_channels_in"] = len(names)
        is_microepi = is_micro is not None

        # ── Aux drop (ECG / DC / markers)
        if is_microepi:
            _n_before = len(names)
            kept_idx = [i for i, nm in enumerate(names) if not _is_non_neural(nm)]
            signals  = signals[:, kept_idx]
            names    = [names[i] for i in kept_idx]
            is_micro = is_micro[kept_idx]
            if len(names) < _n_before:
                print(f"  [{patient_id}] aux drop: {_n_before} → {len(names)} channels")
        else:
            signals, names, *_ = io.filter_aux_channels(signals, names)
            # THE FULL NON-NEURAL TEST FOR EL / HUG TOO (2026-10-02, Lora). filter_aux_channels
            # knows the prefixes MRK / MKR / X / ECG / EX / AUDIO only, so EKG-, EMG-, the HUG
            # photodiode ("photo") and PAT_3975's E1-E4 inputs came through, went into the
            # notch's montage and got ERSP / HFA / Signal figures in 13 patients (the export
            # skipped them, so no cube). _is_non_neural is the test the MicroEPI branch above
            # and the export already use. The photodiode extraction (--pd) loads the raw file
            # itself and is not affected.
            # THE PHOTODIODE STAYS, AS A FIGURES-ONLY CHANNEL (Lora, 2026-10-02): its ERSP and HFA
            # raster are the check that the triggers sit where the screen changed. It is in the
            # four HUG recordings that name it "photo" (PAT_2868, 3390, 3415, 3455). It is not
            # data: no cube (the export skips it), no trial-score rows, no Signal figure, and it
            # is its own "shaft" for the notch, so it decides nothing for a contact.
            _nn = [n for n in names if _is_non_neural(n) and not _is_photodiode(n)]
            if _nn:
                _kp = [i for i, n in enumerate(names) if not (_is_non_neural(n) and not _is_photodiode(n))]
                signals = signals[:, _kp]
                names = [names[i] for i in _kp]
                print(f"  [{patient_id}] aux drop (non-neural): {' '.join(_nn)}")

        # ── EL prefix / grid filter — EL ONLY
        if (not is_microepi) and str(pid_raw).startswith("EL"):
            if pid_raw in cfg.EL_GRID_PATIENTS:
                prefixes = cfg.EL_GRID_KEEP_PREFIXES.get(pid_raw, ())
                keep = [i for i, nm in enumerate(names)
                        if any(str(nm).startswith(p) for p in prefixes)
                        and any(c.isdigit() for c in str(nm))]
            else:
                keep = [i for i, nm in enumerate(names) if ("_" in str(nm) or "-" in str(nm))]
            signals = signals[:, keep]
            names   = [names[i] for i in keep]
            if len(names) == 0:
                print(f"[skip] {patient_id}: no neural channels")
                report["status"] = "no-neural-channels"
                return report
        report["n_channels_neural"] = len(names)

        # ── EXPERIMENT-WINDOW CROP (EL ONLY)
        crop_offset = 0
        _preset = cfg.EL_PRESETS.get(pid_raw) if (not is_microepi and str(pid_raw).startswith("EL")) else None
        if _preset and _preset.get("time_range") and _preset["time_range"][1] > 0:
            t0, t1 = _preset["time_range"]
            s0 = max(0, int(t0 * fs))
            s1 = min(signals.shape[0], int(t1 * fs))
            if s1 > s0 and (s1 - s0) < signals.shape[0]:
                full_s = signals.shape[0] / fs
                signals     = np.ascontiguousarray(signals[s0:s1, :])
                crop_offset = s0
                print(f"  [{patient_id}] cropped to {t0:.0f}s..{t1:.0f}s "
                      f"({(s1-s0)/fs:.1f}s of {full_s:.1f}s)")

        # ── PER-PATIENT CHANNEL-NAME OVERRIDE (EL ONLY)
        if (not is_microepi) and patient_id in getattr(cfg, "STRIP_HEMI_PATIENTS", set()):
            import re as _re
            _hemi_re = _re.compile(r"_(?:[LR])(?=\d)")
            names = [_hemi_re.sub("", str(nm)) for nm in names]
            print(f"  [{patient_id}] stripped _L#/_R# per cfg.STRIP_HEMI_PATIENTS  "
                  f"-> first few: {names[:8]}")

        # ── DROP "UNKNOWN" PARCELLATION CHANNELS (every cohort since 2026-09-23; was EL/PAT
        # only, which left 29 unparcellated MicroEPI contacts in the data)
        if True:
            try:
                unk_idx = set(io.unknown_indices_for_patient(
                    patient_id, names, electrodes_tsv_pattern=_wm_tsv))
            except Exception as _e_unk:
                print(f"[warn] {patient_id}: Unknown-channel lookup failed ({_e_unk}); keeping all")
                unk_idx = set()
            if patient_id in getattr(cfg, "MIXED_GRID_DEPTH_PATIENTS", set()):
                keep_prefixes = cfg.MIXED_GRID_KEEP_PREFIXES.get(patient_id, ())
                if keep_prefixes:
                    protected = {i for i in unk_idx
                                 if any(str(names[i]).startswith(p) for p in keep_prefixes)}
                    if protected:
                        print(f"  [{patient_id}] protected {len(protected)} grid channels "
                              f"from Unknown drop (prefixes={keep_prefixes})")
                    unk_idx -= protected
            if unk_idx:
                report["n_channels_unknown_dropped"] = len(unk_idx)
                keep = [i for i in range(len(names)) if i not in unk_idx]
                signals = signals[:, keep]
                names   = [names[i] for i in keep]
                if is_micro is not None:          # MicroEPI: the micro mask follows the channels
                    is_micro = is_micro[keep]
                print(f"  [{patient_id}] dropped {len(unk_idx)} 'Unknown' channels (no parcellation in TSV)")

        # ── DROP BAD-LISTED CONTACTS (2026-09-28, Lora) - before the reference, the PSD figures
        # and the notch. Until now cfg.bad_channels_manual only kept a contact out of the
        # reference (by name) and out of the ERSP export, so the per-shaft notch still decided
        # each shaft's harmonics and widths on a median that included the bad contacts
        # (EL045's PlaT_L: three of six). Dropped here they are in nothing downstream. The WM
        # ones are still reported in wm_channels_excluded_as_bad through _dropped_bad_keys.
        _bad_norm_drop = {io.normalize_label(b)
                          for b in getattr(cfg, "bad_channels_manual", {}).get(patient_id, [])}
        _dropped_bad = [n for n in names if io.normalize_label(n) in _bad_norm_drop]
        _dropped_bad_keys = [io.alias_label(n, patient_id) for n in _dropped_bad]
        if _dropped_bad:
            keep = [i for i, n in enumerate(names) if io.normalize_label(n) not in _bad_norm_drop]
            signals = signals[:, keep]
            names   = [names[i] for i in keep]
            if is_micro is not None:
                is_micro = is_micro[keep]
            print(f"  [{patient_id}] dropped {len(_dropped_bad)} bad-listed channels before the "
                  f"reference and the notch: {' '.join(_dropped_bad[:12])}"
                  f"{' ...' if len(_dropped_bad) > 12 else ''}")

        # ── REREFERENCING
        if is_microepi:
            pid_str = str(pid_raw)
            preset  = cfg.MICROEPI_MAT_PRESETS.get(pid_str, {})
            # the same WM set as the EL / PAT path (2026-09-22): the TSV rule, minus
            # cfg.WM_NOT_REFERENCE (PAT_6704 THD1/3/4 had stayed in the reference here), in
            # the recording's spelling through cfg.CHANNEL_SHAFT_ALIAS (PAT_6619 OFA / OFP)
            _wm_keys = io.wm_labels_for_patient(patient_id, electrodes_tsv_pattern=_wm_tsv)
            wm_names_raw = [n for n in names if io.alias_label(n, patient_id) in _wm_keys]
            print(f"  [{patient_id}] WM channels from TSV: {len(wm_names_raw)} "
                  f"-> {wm_names_raw[:6]}{'...' if len(wm_names_raw) > 6 else ''}")
            # the bad list stays out of the reference here as it does for EL / PAT (2026-09-21:
            # until now the MicroEPI reference averaged bad-listed contacts in - PAT_6704 had 4)
            _bad_norm_ref = {io.normalize_label(b) for b in getattr(cfg, "bad_channels_manual", {}).get(patient_id, [])}
            try:
                signals, wm_used, wm_excl = mm.apply_wm_reref_selective(
                    signals, names, wm_names_raw, is_micro, apply_wm_to_micros=False,
                    bad_channels_for_ref=[n for n in names if io.normalize_label(n) in _bad_norm_ref])
            except Exception as _e_reref:
                print(f"[error] {patient_id}: macro WM reref failed — {_e_reref}")
                report["status"] = "error-reref"
                report["error"]  = str(_e_reref)
                return report
            reref   = "WM"
            wm_skip = set()
            report["n_wm_used"]                   = len(wm_used)
            report["wm_channels_used"]            = "|".join(sorted(wm_used))
            report["wm_channels_excluded_as_bad"] = "|".join(sorted(set(wm_excl) | {k for k in _dropped_bad_keys if k in _wm_keys}))
            print(f"  [{patient_id}] WM reference: {len(wm_used)} contacts used"
                  + (f", {len(wm_excl)} bad-listed left out: {sorted(wm_excl)}" if wm_excl else ""))
            anchor_name = preset.get("micro_reref_anchor")
            signals, anchor_used = mm.apply_micro_anchor_reref(signals, names, is_micro, anchor_name)
            if anchor_used:
                print(f"  [{patient_id}] micros re-referenced to anchor: {anchor_used}")
            else:
                print(f"  [{patient_id}] micros left raw (no micro_reref_anchor in preset)")
        else:
            is_grid    = str(pid_raw) in getattr(cfg, "EL_GRID_PATIENTS", set())
            wm_all     = io.wm_labels_for_patient(patient_id, electrodes_tsv_pattern=_wm_tsv)
            _bad_norm_for_report = {io.normalize_label(b)
                                    for b in getattr(cfg, "bad_channels_manual", {}).get(patient_id, [])}
            _name_norm = [io.alias_label(n, patient_id) for n in names]   # the key apply_wm_reref matches on (cfg.CHANNEL_SHAFT_ALIAS)
            wm_in_sig  = [n for n in wm_all if n in _name_norm]
            wm_usable  = [n for n in wm_in_sig if n not in _bad_norm_for_report]

            if str(pid_raw) in getattr(cfg, "WHOLE_CAR_PATIENTS", set()):
                # neural channels only (2026-09-30): the aux drop above keeps the HUG
                # photodiode ("photo"), EKG-/EMG- and E1-E4, which the export skips through
                # _is_non_neural but the mean did not - PAT_3415's CAR carried the photodiode
                # step into every contact (a vertical line at 0 % and 50 % of every ERSP)
                _good = [i for i, n in enumerate(names)
                         if io.normalize_label(n) not in _bad_norm_for_report and not _is_non_neural(n)]
                if len(_good) < 3:
                    print(f"[error] {patient_id}: whole-recording CAR needs >= 3 good channels, has {len(_good)}")
                    report["status"] = "error-reref"
                    report["error"]  = f"whole CAR: {len(_good)} good channels"
                    return report
                _ref    = signals[:, _good].astype(float).mean(axis=1, keepdims=True)
                signals = signals.astype(float) - _ref
                reref   = "CAR"
                wm_skip = set()
                print(f"[note] {patient_id}: whole-recording CAR - mean of {len(_good)} channels, "
                      f"{len(names) - len(_good)} bad or non-neural channels left out of the mean: "
                      f"{' '.join(n for i, n in enumerate(names) if i not in set(_good))}")
                report["n_wm_used"]                   = 0
                report["wm_channels_used"]            = f"CAR:all({len(_good)})"
                report["wm_channels_excluded_as_bad"] = "|".join(sorted(n for n in names if io.normalize_label(n) in _bad_norm_for_report))
            elif wm_usable:
                try:
                    signals, reref, wm_skip = apply_wm_reref(
                        signals, names, patient_id, electrodes_tsv_pattern=_wm_tsv)
                    wm_excluded_norm = sorted(set([n for n in wm_in_sig if n in _bad_norm_for_report]
                                                  + [k for k in _dropped_bad_keys if k in wm_all]))
                    report["n_wm_used"]                    = len(wm_usable)
                    report["wm_channels_used"]             = "|".join(sorted(wm_usable))
                    report["wm_channels_excluded_as_bad"]  = "|".join(sorted(wm_excluded_norm))
                except ValueError as _e_reref:
                    print(f"[error] {patient_id}: WM reref failed — {_e_reref}")
                    report["status"] = "error-reref"
                    report["error"]  = str(_e_reref)
                    return report
            elif is_grid:
                if str(pid_raw) in getattr(cfg, "GRID_CAR_PATIENTS", set()):
                    _bad_car = getattr(cfg, "bad_channels_manual", {}).get(patient_id, [])
                    _car_prefixes = cfg.EL_GRID_KEEP_PREFIXES.get(pid_raw, None)
                    signals, car_groups = fe.apply_grid_car(
                        signals, names, group_prefixes=_car_prefixes,
                        bad_channels=_bad_car, min_group=2)
                    reref   = "CAR"
                    wm_skip = set()
                    _summary = ", ".join(f"{k}({len(v)})" for k, v in sorted(car_groups.items()))
                    print(f"[note] {patient_id}: per-grid CAR applied — groups: {_summary}")
                    report["n_wm_used"]                   = 0
                    report["wm_channels_used"]            = "CAR:" + _summary
                    report["wm_channels_excluded_as_bad"] = "|".join(sorted(_bad_car))
                else:
                    print(f"[note] {patient_id}: grid patient with no WM contacts — "
                          f"falling back to reref='NONE' (no rereferencing applied)")
                    reref   = "NONE"
                    wm_skip = set()
                    report["n_wm_used"]                   = 0
                    report["wm_channels_used"]            = ""
                    report["wm_channels_excluded_as_bad"] = ""
                    report["status"]                      = "ok-no-wm-grid"
            else:
                msg = f"no WM channels available for {patient_id} (not a grid patient)"
                print(f"[error] {msg}")
                report["status"] = "error-reref"
                report["error"]  = msg
                return report

        # ── NOTCH SCOPE (cfg.notch_scope): "file" once here, "block" per condition below
        notch_scope = str(getattr(cfg, "notch_scope", "block")).lower()
        notch_audit_rows, peak_rows = [], []
        per_shaft = notch_per_shaft_for(patient_id, pid_raw)
        if notch_scope == "file":
            signals = apply_notch_with_audit(signals, fs, patient_id, pid_raw,
                                             audit=notch_audit_rows, names=names)
            for _r in notch_audit_rows:
                _r.setdefault("condition", "file"); _r.setdefault("block_s0", 0.0)
                _r.setdefault("block_s1", round(signals.shape[0] / fs, 1))
            if per_shaft and RUN_ERSP_PIPELINE and DO_MONTAGE_PSD_PLOTS:
                fe.plot_psd_by_shaft(
                    signals, fs, names,
                    save_root=io.patient_output_dir(run_root_ersp, patient_id, cfg.block_name, "PSD_clean"),
                    patient_id=patient_id, block_name=cfg.block_name, fmax=cfg.fmax,
                    mains_base=getattr(cfg, "mains_base", 50.0), audit=notch_audit_rows)
            if RUN_ERSP_PIPELINE and DO_MONTAGE_PSD_PLOTS:
                fe.plot_psd_overview(
                    signals=signals, fs=fs, names=names,
                    save_root=io.patient_output_dir(run_root_ersp, patient_id, cfg.block_name, "PSD_clean"),
                    patient_id=patient_id, block_name=cfg.block_name,
                    fmax=cfg.fmax, mains_base=getattr(cfg, "mains_base", 50.0), dpi=600)
        else:
            print(f"  [{patient_id}] notch scope: per condition block "
                  f"(pad {getattr(cfg, 'notch_block_pad_s', 10.0):.0f} s)"
                  + (", decided and applied PER SHAFT" if per_shaft else ""))

        report_dir  = io.patient_output_dir(run_root_ersp, patient_id, cfg.block_name, "Report")
        report_path = os.path.join(report_dir, f"{patient_id}_IQR.tsv")
        all_trials = {}
        cond_groups = tr.collect_trials(prep_dir, fs, outlier_method="IQR",
                                        iqr_k=cfg.iqr_k, report_path=report_path,
                                        patient_id=patient_id, max_post_s=cfg.max_post_s,
                                        min_stim_s=cfg.min_stim_s, min_post_s=cfg.min_post_s,
                                        all_out=all_trials,
                                        bad_spans=getattr(cfg, "bad_time_spans", {}).get(patient_id, []),
                                        bad_spans_pre_s=abs(float(cfg.baseline_w[0])))
        if not cond_groups:
            print(f"[skip] {patient_id}: no trials")
            report["status"] = "ok-no-trials"
            return report

        if crop_offset > 0:
            rebased = {}
            n_samples = signals.shape[0]
            for cond, (on, off, tend) in cond_groups.items():
                on   = np.asarray(on)   - crop_offset
                off  = np.asarray(off)  - crop_offset
                tend = np.asarray(tend) - crop_offset
                ok = (on >= 0) & (tend <= n_samples)
                if (~ok).any():
                    print(f"  [{patient_id} | {cond}] dropped {(~ok).sum()} trials outside crop window")
                rebased[cond] = (on[ok], off[ok], tend[ok])
            cond_groups = rebased

        if RUN_ERSP_PIPELINE:
            if WRITE_MONTAGE:
                tr.plot_montage_overview(signals=signals, fs=fs, names=names,
                                         cond_groups=cond_groups, save_dir=report_dir, patient_id=patient_id)
            ersp_root = io.patient_output_dir(run_root_ersp, patient_id, cfg.block_name, "ERSP")
            hg_root   = io.patient_output_dir(run_root_ersp, patient_id, cfg.block_name, cfg.HFA_DIR)
            sig_root  = io.patient_output_dir(run_root_ersp, patient_id, cfg.block_name,
                                              getattr(cfg, "SIGNAL_DIR", "Signal"))

        if RUN_CLUSTER_EXPORT:
            mat_root = _ensure(io.patient_output_dir(run_root_raw, patient_id, cfg.block_name, "ERSP_matrix"))
            img_root = _ensure(io.patient_output_dir(run_root_raw, patient_id, cfg.block_name, "ERSP_clean"))
            _bad_norm = {io.normalize_label(b)
                         for b in getattr(cfg, "bad_channels_manual", {}).get(patient_id, [])}
            skip      = set(n for n in names if _is_non_neural(n))
            skip     |= {n for n in names if io.normalize_label(n) in _bad_norm}
            skip     |= wm_skip

        micro_names_set = set()
        if is_microepi and not INCLUDE_MICROEPI_MICROS:
            micro_names_set = {names[i] for i in range(len(names)) if is_micro[i]}

        for cond, (onsets, offsets, trial_ends) in cond_groups.items():
            if RUN_ERSP_PIPELINE:
                ersp_dir = _ensure(os.path.join(ersp_root, cond))
                hg_dir   = _ensure(os.path.join(hg_root, cond))
                sig_dir  = os.path.join(sig_root, cond)       # created by the plot when it writes
            if RUN_CLUSTER_EXPORT:
                out_mat  = _ensure(os.path.join(mat_root, cond))
                out_half = os.path.join(io.patient_output_dir(run_root_raw, patient_id,
                                                              cfg.block_name, "ERSP_halves"), cond)
                out_png  = _ensure(os.path.join(img_root, cond))

            # ── PER-BLOCK CLEANING (notch_scope == "block")
            sig_c, on_c, off_c, te_c = signals, onsets, offsets, trial_ends
            if notch_scope != "file":
                _pad = float(getattr(cfg, "notch_block_pad_s", 10.0))
                b0, b1 = fe.block_window(onsets, trial_ends, signals.shape[0], fs, pad_s=_pad)
                sig_c = np.ascontiguousarray(signals[b0:b1, :])
                on_c  = np.asarray(onsets) - b0
                off_c = (np.asarray(offsets) - b0) if (offsets is not None and len(offsets)) else offsets
                te_c  = (np.asarray(trial_ends) - b0) if trial_ends is not None else None
                print(f"  [{patient_id} | {cond}] block {b0/fs:.0f}s..{b1/fs:.0f}s "
                      f"({(b1-b0)/fs:.0f} s, {len(on_c)} trials)")
                _mb = getattr(cfg, "mains_base", 50.0)
                _xb = tuple(getattr(cfg, "notch_extra_bases", {}).get(patient_id, ()))
                if RUN_ERSP_PIPELINE and DO_MONTAGE_PSD_PLOTS:
                    fe.plot_psd_overview(
                        signals=sig_c, fs=fs, names=names,
                        save_root=io.patient_output_dir(run_root_ersp, patient_id, cfg.block_name,
                                                        os.path.join("PSD_raw", cond)),
                        patient_id=patient_id, block_name=f"{cfg.block_name} {cond} (before notch)",
                        fmax=cfg.fmax, mains_base=_mb, dpi=600)
                if per_shaft and RUN_ERSP_PIPELINE and DO_MONTAGE_PSD_PLOTS:
                    fe.plot_psd_by_shaft(
                        sig_c, fs, names,
                        save_root=io.patient_output_dir(run_root_ersp, patient_id, cfg.block_name,
                                                        os.path.join("PSD_raw", cond)),
                        patient_id=patient_id, block_name=f"{cfg.block_name} {cond} (before notch)",
                        fmax=cfg.fmax, mains_base=_mb)
                _audit = []
                sig_c = apply_notch_with_audit(sig_c, fs, patient_id, pid_raw, audit=_audit, names=names)
                for _r in _audit:
                    notch_audit_rows.append(dict(condition=cond, block_s0=round(b0 / fs, 1),
                                                 block_s1=round(b1 / fs, 1), **_r))
                if per_shaft:
                    _pk = fe.unexplained_peaks_by_shaft(
                        sig_c, fs, names, bases=(_mb, *_xb), fmax=cfg.fmax,
                        z_thresh=getattr(cfg, "notch_peak_z_thresh", 3.0))
                else:
                    _pk = fe.unexplained_peaks(sig_c, fs, bases=(_mb, *_xb), fmax=cfg.fmax,
                                               z_thresh=getattr(cfg, "notch_peak_z_thresh", 3.0))
                    _pk.insert(0, "shaft", "all")
                for _r in _pk.to_dict("records"):
                    peak_rows.append(dict(condition=cond, **_r))
                if per_shaft and RUN_ERSP_PIPELINE and DO_MONTAGE_PSD_PLOTS:
                    fe.plot_psd_by_shaft(
                        sig_c, fs, names,
                        save_root=io.patient_output_dir(run_root_ersp, patient_id, cfg.block_name,
                                                        os.path.join("PSD_clean", cond)),
                        patient_id=patient_id, block_name=f"{cfg.block_name} {cond} (after notch)",
                        fmax=cfg.fmax, mains_base=_mb, audit=_audit, peaks=_pk)
                _off = _pk[~_pk.on_comb].freq_hz.tolist()
                _fit = fe.fit_comb(_off)
                _n_notched = sum(1 for _r in _audit if _r["notched"])
                print(f"  [{patient_id} | {cond}] notched {_n_notched} of {len(_audit)} candidates"
                      + (f" over {len(set(_r['shaft'] for _r in _audit))} shafts" if per_shaft else "")
                      + "; "
                      f"{int(_pk.on_comb.sum())} comb peaks remain, {len(_off)} off-comb peaks"
                      + (f" - best comb fit {_fit[0][0]:g} Hz ({_fit[0][2]}/{len(_off)})" if _fit else ""))
                if RUN_ERSP_PIPELINE and DO_MONTAGE_PSD_PLOTS:
                    fe.plot_psd_overview(
                        signals=sig_c, fs=fs, names=names,
                        save_root=io.patient_output_dir(run_root_ersp, patient_id, cfg.block_name,
                                                        os.path.join("PSD_clean", cond)),
                        patient_id=patient_id, block_name=f"{cfg.block_name} {cond} (after notch)",
                        fmax=cfg.fmax, mains_base=_mb, dpi=600)

            # ---- the whole trial table, in sig_c coordinates
            _hg_all = None
            _rec = all_trials.get(cond)
            if _rec is not None and len(on_c):
                _shift = int(crop_offset) + (int(np.asarray(onsets)[0]) - int(np.asarray(on_c)[0]))
                _on_a = np.asarray(_rec["on"]) - _shift
                _off_a = np.asarray(_rec["off"]) - _shift
                _te_a = np.asarray(_rec["tend"]) - _shift
                _in = (_on_a >= 0) & (_te_a <= sig_c.shape[0])
                _kept_a = np.asarray(_rec["keep"]) & _in
                if int(_kept_a.sum()) == len(on_c):
                    _hg_all = dict(on=_on_a[_in], off=_off_a[_in], te=_te_a[_in],
                                   reason=np.asarray(_rec["reason"], dtype=object)[_in],
                                   keep_pos=np.flatnonzero(np.asarray(_rec["keep"])[_in]))
                    _n_out = int((~_in).sum())
                    if _n_out:
                        print(f"  [{patient_id} | {cond}] {_n_out} removed trials fall "
                              f"outside the block and cannot be drawn")
                else:
                    print(f"  [{patient_id} | {cond}] trial tables disagree "
                          f"({int(_kept_a.sum())} vs {len(on_c)}); the HG figure shows the kept trials only")
            print(f"  {cond}: ", end="", flush=True)
            _score_rows = []
            # the Signal figures are drawn against a µV scale; the .edf patients arrive in volts
            _uv_scale, _uv_sd = fe.microvolt_scale(sig_c, fs)
            if _uv_scale != 1.0:
                print(f"[signal in volts: robust SD {_uv_sd:.2e}; Signal figures x{_uv_scale:g}] ", end="", flush=True)
            for ci, chan_name in enumerate(names):
                if chan_name in wm_skip: continue
                if chan_name in micro_names_set: continue
                if RUN_CLUSTER_EXPORT and chan_name in skip and not RUN_ERSP_PIPELINE: continue

                res = fe.compute_ersp(
                    signals=sig_c, fs=fs, onsets=on_c, offsets=off_c, channel_idx=ci,
                    trial_ends=te_c, mode=cfg.mode, time_window=cfg.time_window,
                    params=ersp_params)

                if RUN_ERSP_PIPELINE and WRITE_ERSP_PLOTS:
                    fe.plot_ersp(res, patient_id=patient_id, condition=cond,
                                 reref_type=reref, chan_name=chan_name,
                                 save_dir=ersp_dir, params=ersp_params,
                                 plot_title=False, save_sidecar=False)
                if RUN_ERSP_PIPELINE and WRITE_HG_PLOTS:
                    fe.plot_hg_trials(
                        signals=sig_c, fs=fs,
                        onsets=(_hg_all["on"] if _hg_all is not None else on_c),
                        offsets=(_hg_all["off"] if _hg_all is not None else off_c),
                        channel_idx=ci,
                        chan_name=chan_name, patient_id=patient_id, condition=cond, reref_type=reref,
                        time_window=cfg.time_window, baseline_w=cfg.baseline_w,
                        hg_band=cfg.hg_band, smooth_ms=cfg.hg_smooth_ms,
                        rejected_trials=(
                            [int(_hg_all["keep_pos"][i]) for i in res.get("dropped_trials", [])]
                            if _hg_all is not None else res.get("dropped_trials")),
                        exclude_reasons=(_hg_all["reason"] if _hg_all is not None else None),
                        trial_z=_hg_trial_z(res, _hg_all),
                        vmin=cfg.hg_vmin, vmax=cfg.hg_vmax,
                        save_dir=hg_dir, add_separators=False, sort_ascending=True,
                        trial_end_indices=(_hg_all["te"] if _hg_all is not None else te_c),
                        sort_by="stim")
                # the HFA raster's twin (2026-10-02): the same trials, order, marks and rejected
                # rows, each row the broadband cleaned voltage as a trace, one µV scale for the cohort
                if RUN_ERSP_PIPELINE and WRITE_SIGNAL_PLOTS and not _is_photodiode(chan_name):
                    fe.plot_signal_trials(
                        signals=sig_c, fs=fs,
                        onsets=(_hg_all["on"] if _hg_all is not None else on_c),
                        offsets=(_hg_all["off"] if _hg_all is not None else off_c),
                        channel_idx=ci,
                        chan_name=chan_name, patient_id=patient_id, condition=cond, reref_type=reref,
                        time_window=cfg.time_window, baseline_w=cfg.baseline_w,
                        uv_per_row=getattr(cfg, "signal_plot_uv_per_row", 200.0),
                        unit_scale=_uv_scale,
                        n_rows_fixed=getattr(cfg, "signal_plot_rows", 58),
                        rejected_trials=(
                            [int(_hg_all["keep_pos"][i]) for i in res.get("dropped_trials", [])]
                            if _hg_all is not None else res.get("dropped_trials")),
                        exclude_reasons=(_hg_all["reason"] if _hg_all is not None else None),
                        trial_z=_hg_trial_z(res, _hg_all),
                        save_dir=sig_dir, sort_ascending=True,
                        trial_end_indices=(_hg_all["te"] if _hg_all is not None else te_c),
                        sort_by="stim")

                if RUN_CLUSTER_EXPORT and chan_name not in skip:
                    A = np.array(res["avg_db"], float)
                    if np.isnan(A).any():
                        print(f"[warn] {patient_id} {cond} {chan_name}: {int(np.isnan(A).sum())} NaNs → filling")
                        fill_nans_nearest(A)
                    mode_tag = "_TN" if str(res["meta"]["mode"]).upper() == "TN" else ""
                    stem = f"{patient_id}_{cond}_{reref}_ERSP_{chan_name}{mode_tag}"
                    if WRITE_CUBES:
                        np.save(os.path.join(out_mat, f"{stem}.npy"), A)
                    # split-half cubes for the reliability gate, in ERSP_halves/ (see the notebook)
                    for _h, _key in (("half1", "avg_db_h1"), ("half2", "avg_db_h2")):
                        _H = res.get(_key)
                        if _H is None or not WRITE_HALVES:
                            continue
                        _ensure(out_half)
                        np.save(os.path.join(out_half, f"{stem}_{_h}.npy"), np.asarray(_H, float))
                    if WRITE_CLEAN_PNG:
                        save_clean_png(A, vmin=ersp_params.vmin, vmax=ersp_params.vmax,
                                       path_png=os.path.join(out_png, f"{stem}_CLEAN.png"))
                    del A

                # ONE ROW PER (CHANNEL, TRIAL). compute_ersp already scores every trial;
                # nothing here re-derives it, so the numbers written are the ones the
                # rejection rule would compare against. They are all-NaN unless a
                # trial_reject_* threshold is set, which is why 145 sets an
                # unreachable one - it turns the scoring on without dropping anything.
                if EXPORT_TRIAL_SCORES and not _is_non_neural(chan_name):
                    _sc = res.get("trial_scores") or []
                    _sh = res.get("trial_scores_hg") or []
                    _drop = set(res.get("dropped_trials") or [])
                    # TWO TRIAL NUMBERINGS, and they are not the same one. `trial` counts
                    # only the trials that reached the ERSP; `trial_label` is the number
                    # the HG figure prints beside the row, which counts the WHOLE table
                    # including the trials a filter removed first. On PAT_3455 picture
                    # that is 47 against 53. Both are written, so a trial the sweep names
                    # can be found on the figure without arithmetic.
                    _kp = (_hg_all["keep_pos"] if _hg_all is not None else None)
                    for _t in range(len(_sc)):
                        _orig = int(_kp[_t]) if (_kp is not None and _t < len(_kp)) else _t
                        _score_rows.append(dict(
                            patient=patient_id, condition=cond, channel=chan_name,
                            trial=_t, trial_label=_orig + 1,
                            score_map=_sc[_t],
                            score_hg=(_sh[_t] if _t < len(_sh) else float("nan")),
                            dropped_by_current_rule=int(_t in _drop)))
                del res
            print(" done")

            if EXPORT_TRIAL_SCORES and _score_rows:
                _sdir = _ensure(io.patient_output_dir(run_root_ersp, patient_id,
                                                      cfg.block_name, "TrialScores"))
                _sp = os.path.join(_sdir, f"{patient_id}_{cond}_trial_scores.tsv")
                pd.DataFrame(_score_rows).to_csv(_sp, sep="	", index=False)
                _n_ch = len(set(r["channel"] for r in _score_rows))
                _n_tr = len(set(r["trial"] for r in _score_rows))
                print(f"  [{patient_id} | {cond}] trial scores -> {os.path.basename(_sp)} "
                      f"({_n_ch} channels x {_n_tr} trials)")

        # ── PER-BLOCK CLEANING REPORT
        if notch_audit_rows:
            pd.DataFrame(notch_audit_rows).to_csv(
                os.path.join(report_dir, f"{patient_id}_notch_audit.tsv"), sep="\t", index=False)
        if notch_scope != "file":
            pd.DataFrame(peak_rows, columns=["condition", "shaft", "freq_hz", "z", "peak_db", "above_db", "on_comb"]).to_csv(
                os.path.join(report_dir, f"{patient_id}_unexplained_peaks.tsv"), sep="\t", index=False)
            print(f"  [{patient_id}] notch audit -> {patient_id}_notch_audit.tsv, "
                  f"{patient_id}_unexplained_peaks.tsv ({len(peak_rows)} peaks)")

        if not report["status"]:
            report["status"] = "ok"
        return report
    except Exception as e:
        print(f"[error] {pid_raw}: {e}")
        report["status"] = report["status"] or "error-other"
        report["error"]  = str(e)
        return report


def merge_report(row):
    """outputs/03_ERSP/wm_reref_report.tsv: one row per patient, this run's row
    replacing the patient's previous one, the others kept (the notebook rewrote the whole
    file per batch; one patient per process needs a merge)."""
    out_tsv = os.path.join(run_root_raw, "wm_reref_report.tsv")
    os.makedirs(os.path.dirname(out_tsv) or ".", exist_ok=True)
    new = pd.DataFrame([row])
    if os.path.exists(out_tsv):
        try:
            old = pd.read_csv(out_tsv, sep="\t")
            old = old[old["patient_id"].astype(str) != str(row["patient_id"])]
            new = pd.concat([old, new], ignore_index=True)
        except Exception as e:
            print(f"  [warn] could not merge into {out_tsv}: {e}")
    new.to_csv(out_tsv, sep="\t", index=False)
    print(f"  saved -> {out_tsv}")


def print_report_line(r):
    status_tag = {
        "ok": "  OK   ", "ok-no-trials": "OK-NOTR", "ok-no-wm-grid": "OK-GRID",
        "no-neural-channels": "NO-CHAN", "error-reref": "REREF✗ ", "error-load": "LOAD✗  ",
        "error-other": "ERR✗   ", "": "??     ",
    }.get(r["status"], r["status"])
    line = (f"  {status_tag}  {r['patient_id']:<14}  in={r['n_channels_in']:>4}  "
            f"neural={r['n_channels_neural']:>4}  unknown_dropped={r['n_channels_unknown_dropped']:>3}  "
            f"used={r['n_channels_used']:>4}  wm={r['n_wm_used']:>3}")
    if r["error"]:
        line += f"  err='{str(r['error'])[:60]}'"
    print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--patient", required=True, nargs="+",
                    help="raw id(s) as config.py spells them: EL030, PAT_3455, G-01, PAT_6684 ...")
    ap.add_argument("--pd", action="store_true", help="run the photodiode trial extraction first (rewrites prep0)")
    ap.add_argument("--no-qc", action="store_true", help="skip the QC figures (ERSP/HFA/PSD/montage); cubes only")
    ap.add_argument("--no-export", action="store_true", help="skip the cubes, halves and CLEAN PNGs")
    ap.add_argument("--no-psd", action="store_true", help="keep the QC but skip the dpi-600 PSD montages")
    ap.add_argument("--macros-only", action="store_true", help="MicroEPI patients: skip the micro wires")
    a = ap.parse_args()

    RUN_ERSP_PIPELINE  = not a.no_qc
    RUN_CLUSTER_EXPORT = not a.no_export
    DO_MONTAGE_PSD_PLOTS = not a.no_psd
    INCLUDE_MICROEPI_MICROS = not a.macros_only

    log_dir = os.path.join(run_root_ersp, "logs")
    os.makedirs(log_dir, exist_ok=True)
    for pid_raw in a.patient:
        # config keys are ints for the old HUG spelling (3455) and strings otherwise
        pid = int(pid_raw) if str(pid_raw).isdigit() else str(pid_raw)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"{out_id_for(pid)}_{stamp}.log")
        tee = _Tee(log_path)
        sys.stdout = tee
        try:
            print(f"[140] {out_id_for(pid)}  ({pid})  {stamp}   fmax {cfg.fmax:g} Hz   "
                  f"qc={'on' if RUN_ERSP_PIPELINE else 'off'} export={'on' if RUN_CLUSTER_EXPORT else 'off'} "
                  f"psd={'on' if DO_MONTAGE_PSD_PLOTS else 'off'}   log -> {log_path}")
            if a.pd:
                run_pd_extraction(pid)
                print("\n[PD extraction done]")
            if not RUN_ERSP_PIPELINE and not RUN_CLUSTER_EXPORT:
                print("[skip] both pipelines disabled")
                continue
            print("\n")
            print(pid)
            row = process_patient(pid, RUN_ERSP_PIPELINE, RUN_CLUSTER_EXPORT,
                                  DO_MONTAGE_PSD_PLOTS, INCLUDE_MICROEPI_MICROS)
            plt.close("all")
            gc.collect()
            print("\nPipeline done.")
            print("\n" + "=" * 72)
            print("WM REREFERENCING REPORT (this patient)")
            print("=" * 72)
            print_report_line(row)
            merge_report(row)
        finally:
            sys.stdout = tee._o
            tee._f.close()


if __name__ == "__main__":
    main()
