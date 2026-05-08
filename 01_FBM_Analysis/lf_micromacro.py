"""
lf_micromacro.py
================

Helper functions for the MicroEPI micro/macro ERSP comparison pipeline.

Used by:
    01_FBM_Analysis/scripts/11_MicroMacroComparison_Analysis.ipynb

Scope:
    Only MicroEPI-specific logic lives here. Shared logic is reused from:
        - lf_io_utils.py
        - lf_trials.py
        - lf_ersp.py
        - LFfunctions_PDextract.py (photodiode + TSV saving)

Style:
    Prefer list/dict comprehensions where they read naturally; keep function
    bodies short but clear. No forced one-liners.
"""
from __future__ import annotations
import os, re, glob
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# .mat loader (h5py — MATLAB v7.3 files are large)
# -----------------------------------------------------------------------------
_CHAN_NAME_FIELDS = ("name", "label", "Name", "Label")  # tried in order


def _h5_refs_to_strings(f, refs_dataset):
    """Convert an HDF5 dataset of object refs (each ref -> char-array) to a list[str]."""
    refs = refs_dataset[()].ravel()
    out = []
    for r in refs:
        chars = f[r][()].ravel()
        out.append("".join(chr(int(c)) for c in chars if int(c) != 0))
    return out


def _read_channel_struct_names(f, struct_key):
    """
    Extract channel names from a MATLAB struct array saved as HDF5.

    Tries the common field names in order ('name', 'label', then capitalized
    variants). Different MicroEPI export scripts use different conventions:
    older exports use `name`, newer ones use `label`. Raises a clear error
    listing the available fields if none of them work.
    """
    grp = f[struct_key]
    available = list(grp.keys()) if hasattr(grp, "keys") else []
    for field in _CHAN_NAME_FIELDS:
        if field in available:
            try:
                return _h5_refs_to_strings(f, grp[field])
            except Exception:
                continue
    raise KeyError(
        f"Could not extract channel names from '{struct_key}': none of "
        f"{_CHAN_NAME_FIELDS} found. Available fields: {available}"
    )


def _scipy_struct_names(struct_array, fields=_CHAN_NAME_FIELDS):
    """
    Extract channel names from a scipy.io.loadmat struct array (non-HDF5 .mat).

    scipy returns MATLAB structs as numpy structured arrays of shape (1, N) or
    (N, 1). Names are nested one or two levels deep depending on save options.
    """
    s = struct_array
    if hasattr(s, "dtype") and s.dtype.names:
        for field in fields:
            if field in s.dtype.names:
                vals = s[field].ravel()
                names = []
                for v in vals:
                    # Could be a 1-element array containing a string, or directly a string
                    if hasattr(v, "ravel") and v.size:
                        v = v.ravel()[0]
                    if isinstance(v, bytes):
                        v = v.decode()
                    names.append(str(v))
                return names
        raise KeyError(f"scipy struct fields {s.dtype.names} contain none of {fields}")
    raise TypeError(f"scipy struct has no named fields (dtype={getattr(s, 'dtype', '?')})")


def load_microepi_mat(path):
    """
    Load one MicroEPI export .mat — handles both v7.3 (HDF5) and v7-or-earlier
    (legacy MATLAB binary) formats.

    Some exports (`*_export_Labs_ph.mat` — no `microdown` suffix) only contain
    macros + photodiode and have no `dataMicroDown` / `chansMicro` fields. In
    that case `data_micro` is returned as a (n_samples, 0) array and
    `chans_micro` as an empty list, so the rest of the pipeline can still
    process the macros uniformly.

    Returns
    -------
    dict with:
        'data_ecog'   : (n_samples, n_macro) float32
        'data_micro'  : (n_samples, n_micro) float32   — may be (N, 0) if absent
        'chans_ecog'  : list[str]
        'chans_micro' : list[str]                       — may be []
        'photodiode'  : (n_samples,) float32
        'fs'          : 2048.0
    """
    # Try v7.3 (HDF5) first; fall back to v7-or-earlier via scipy.io
    try:
        return _load_microepi_mat_v73(path)
    except OSError as e:
        if "file signature not found" not in str(e):
            raise  # different OSError, don't mask it
        return _load_microepi_mat_v7(path)


def _load_microepi_mat_v73(path):
    """v7.3 (HDF5) MicroEPI MAT loader."""
    with h5py.File(path, "r") as f:
        data_ecog  = np.asarray(f["dataEcog"][()],   dtype=np.float32)
        photodiode = np.asarray(f["photodiode"][()], dtype=np.float32).ravel()
        if data_ecog.shape[0] < data_ecog.shape[1]:
            data_ecog = data_ecog.T
        chans_ecog = _read_channel_struct_names(f, "chansEcog")

        if "dataMicroDown" in f and "chansMicro" in f:
            data_micro = np.asarray(f["dataMicroDown"][()], dtype=np.float32)
            if data_micro.shape[0] < data_micro.shape[1]:
                data_micro = data_micro.T
            chans_micro = _read_channel_struct_names(f, "chansMicro")
        else:
            data_micro  = np.zeros((data_ecog.shape[0], 0), dtype=np.float32)
            chans_micro = []

    return dict(
        data_ecog=data_ecog, data_micro=data_micro,
        chans_ecog=chans_ecog, chans_micro=chans_micro,
        photodiode=photodiode, fs=2048.0,
    )


def _load_microepi_mat_v7(path):
    """Legacy v7 / v6 / v4 MicroEPI MAT loader via scipy.io.loadmat."""
    from scipy.io import loadmat
    md = loadmat(path, squeeze_me=False, struct_as_record=True)

    data_ecog  = np.asarray(md["dataEcog"],   dtype=np.float32)
    photodiode = np.asarray(md["photodiode"], dtype=np.float32).ravel()
    if data_ecog.shape[0] < data_ecog.shape[1]:
        data_ecog = data_ecog.T
    chans_ecog = _scipy_struct_names(md["chansEcog"])

    if "dataMicroDown" in md and "chansMicro" in md:
        data_micro = np.asarray(md["dataMicroDown"], dtype=np.float32)
        if data_micro.shape[0] < data_micro.shape[1]:
            data_micro = data_micro.T
        chans_micro = _scipy_struct_names(md["chansMicro"])
    else:
        data_micro  = np.zeros((data_ecog.shape[0], 0), dtype=np.float32)
        chans_micro = []

    return dict(
        data_ecog=data_ecog, data_micro=data_micro,
        chans_ecog=chans_ecog, chans_micro=chans_micro,
        photodiode=photodiode, fs=2048.0,
    )


def load_and_concatenate_mats(data_dir, mat_files):
    """
    Load every .mat in `mat_files` and concatenate along the sample axis.
    Channel lists come from file 1 and are asserted consistent across files.
    """
    loaded = [load_microepi_mat(os.path.join(data_dir, fn)) for fn in mat_files]

    chans_ecog, chans_micro = loaded[0]["chans_ecog"], loaded[0]["chans_micro"]
    for i, d in enumerate(loaded[1:], start=1):
        assert d["chans_ecog"]  == chans_ecog,  f"chansEcog mismatch in {mat_files[i]}"
        assert d["chans_micro"] == chans_micro, f"chansMicro mismatch in {mat_files[i]}"

    return dict(
        data_ecog  = np.concatenate([d["data_ecog"]  for d in loaded], axis=0),
        data_micro = np.concatenate([d["data_micro"] for d in loaded], axis=0),
        photodiode = np.concatenate([d["photodiode"] for d in loaded], axis=0),
        chans_ecog=chans_ecog, chans_micro=chans_micro, fs=loaded[0]["fs"],
    )


# -----------------------------------------------------------------------------
# WM channel derivation from BIDS electrodes TSV (canonical home: lf_io_utils)
# Re-exported here so notebook 11 keeps working (it calls `mm.derive_...`).
# -----------------------------------------------------------------------------
# from lf_io_utils import derive_wm_channels_from_electrodes_tsv  # noqa: F401


# -----------------------------------------------------------------------------
# Channel pairing: micro shaft -> first macro contact
# -----------------------------------------------------------------------------
_MICRO_RE = re.compile(r"^([A-Za-z]+)m(\d+)$")

def pair_micro_to_macro(chans_micro, chans_ecog):
    """
    Pair micros ('<prefix>m<N>') to the first macro contact ('<prefix>1').

    Returns
    -------
    dict: { shaft_prefix: { 'macro': '<prefix>1', 'micros': [sorted micro names] } }
    Shafts whose expected macro is missing from chans_ecog are skipped with a warning.
    """
    ecog_set = set(chans_ecog)
    shafts = {}
    for nm in chans_micro:
        m = _MICRO_RE.match(str(nm))
        if not m:
            continue
        prefix, num = m.group(1), int(m.group(2))
        shafts.setdefault(prefix, []).append((num, str(nm)))

    out = {}
    for prefix, items in shafts.items():
        macro = f"{prefix}1"
        if macro not in ecog_set:
            print(f"[pair_micro_to_macro] skip shaft '{prefix}': macro '{macro}' not in chansEcog")
            continue
        out[prefix] = {"macro": macro, "micros": [nm for _, nm in sorted(items)]}
    return out


# -----------------------------------------------------------------------------
# Tetrode grouping
# -----------------------------------------------------------------------------
def group_into_tetrodes(micro_list):
    """
    Chunk an ordered list of micro channel names into groups of 4.
    Trailing micros that don't fill a tetrode are dropped with a warning.
    """
    n_full = (len(micro_list) // 4) * 4
    if n_full != len(micro_list):
        print(f"[group_into_tetrodes] dropping {len(micro_list) - n_full} trailing micros "
              f"(len={len(micro_list)} not divisible by 4)")
    return [micro_list[i:i+4] for i in range(0, n_full, 4)]


# -----------------------------------------------------------------------------
# Photodiode event extraction (wraps LFfunctions_PDextract)
# -----------------------------------------------------------------------------
def extract_events_from_photodiode(photodiode, fs, *, trig_name="photodiode", time_range=(0, -1),
                                    do_plot=True, trial_ids=None, invalid_trials=None,
                                    fake_trials=None, extra_table_path=None, **pd_kwargs):
    """
    Run square-wave photodiode detection from a 1-D photodiode trace.
    Wraps LFfunctions_PDextract.get_trigger_indexes_photodiode.
    """
    from LFfunctions_PDextract import get_trigger_indexes_photodiode

    pd_2d = np.asarray(photodiode, dtype=np.float32).reshape(-1, 1)
    out = get_trigger_indexes_photodiode(
        raw_signals=pd_2d, sampling_rate=float(fs),
        channel_names=[trig_name], trig_name=trig_name,
        time_range=time_range, do_plot=do_plot,
        trial_ids=trial_ids,
        invalid_trials=invalid_trials if invalid_trials is not None else [],
        fake_trials=fake_trials if fake_trials is not None else [],
        extra_table_path=extra_table_path,
        **pd_kwargs,
    )
    on_abs, off_abs = out[0], out[1]
    return np.asarray(on_abs, dtype=np.int64), np.asarray(off_abs, dtype=np.int64)


# -----------------------------------------------------------------------------
# 140-pipeline integration: load only the macro signals from a MicroEPI .mat
# preset, in the same shape that `lf_io_utils.load_first_raw_in_dir` returns.
# Used by 140 for G-04 / G-05 / G-06 (PAT_6704 / PAT_6684 / PAT_6854).
# -----------------------------------------------------------------------------
def load_microepi_macros_for_pipeline(pid_raw, presets):
    """
    Load only macro (`dataEcog`) signals for one MicroEPI .mat-pipeline patient.
    If micro data is absent (e.g. G-01/02/03 plain .mat exports), that is
    silently ignored — only macros are needed for the 140 pipeline.
    """
    cfg = presets[pid_raw]
    d = load_and_concatenate_mats(cfg["data_dir"], cfg["mat_files"])
    return d["data_ecog"], list(d["chans_ecog"]), d["fs"], d["photodiode"]

def extract_microepi_pd_to_prep0(pid_raw, presets, prep_dir, *, do_plot=True):
    """
    Run photodiode event extraction on a MicroEPI .mat preset and write
    per-condition timing TSVs into `prep_dir` (same format that
    `lf_trials.collect_trials` expects).

    Now reads flip, time_range, and trial_ids from the preset dict.
    Pass do_plot=False to suppress the photodiode figure.

    Returns (prep_dir, fs).
    """
    from LFfunctions_PDextract import _read_trial_table, save_onsets_offsets_by_condition

    cfg     = presets[pid_raw]
    pat_id  = cfg.get("pat_name", pid_raw)

    d  = load_and_concatenate_mats(cfg["data_dir"], cfg["mat_files"])
    fs = d["fs"]

    # --- photodiode extraction with per-patient params ---
    on_abs, off_abs = extract_events_from_photodiode(
        d["photodiode"], fs,
        trig_name         = cfg.get("trig", "photodiode"),
        time_range        = cfg.get("time_range", (0, -1)),
        flip_trigs        = cfg.get("flip", False),
        do_plot           = do_plot,
        trial_ids         = cfg.get("trial_ids", None),
        invalid_trials    = cfg.get("invalid_trials", []),
        fake_trials       = cfg.get("fake_trials", []),
        extra_table_path  = os.path.join(cfg["data_dir"], cfg["tsv_file"]) if cfg.get("tsv_file") else None,
    )
    print(f"  [{pat_id}] photodiode: {len(on_abs)} onsets, {len(off_abs)} offsets")

    # --- behavioural TSV ---
    beh_path = os.path.join(cfg["data_dir"], cfg["tsv_file"])
    beh  = _read_trial_table(beh_path)
    dfl  = beh["raw_df"].rename(columns=str.lower)

    def _pick(cols):
        return next((dfl[c].astype(str).to_numpy() for c in cols if c in dfl), None)

    condition_name  = _pick(["category", "blockname"])
    resp_accuracy   = _pick(["response_type", "responseaccuracy"])
    trial_idx_col   = _pick(["exemplar", "stimnumber"])

    # Use preset trial_ids if provided, otherwise derive from TSV
    preset_trial_ids = cfg.get("trial_ids", None)
    if preset_trial_ids is not None:
        trial_ids = preset_trial_ids
    else:
        short = {"picture": "pict", "auditory": "audi", "reading": "read"}
        trial_ids = [short.get(str(x).lower().split("_")[0], str(x).lower())
                     for x in (condition_name if condition_name is not None else [])]

    # # honour invalid_trials by zeroing those entries (same convention as PAT loop)
    # invalid = cfg.get("invalid_trials", [])
    # if invalid and len(trial_ids) > 0:
    #     trial_ids = list(trial_ids)
    #     for idx in invalid:
    #         if idx < len(trial_ids):
    #             trial_ids[idx] = "invalid"

    # --- compute trial_end_abs from response_time in behavioral TSV ---
    _rt_col = next((c for c in dfl.columns if c in ("response_time", "responsetime", "rt")), None)
    if _rt_col is not None:
        _rt = pd.to_numeric(dfl[_rt_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        _trial_end_abs = (off_abs + (_rt[:len(off_abs)] * fs)).astype(np.int64)
        _trial_end_abs = np.maximum(_trial_end_abs, off_abs + 1)
    else:
        _trial_end_abs = (off_abs + int(fs * 2.0)).astype(np.int64)  # 2 s fallback

    os.makedirs(prep_dir, exist_ok=True)
    save_onsets_offsets_by_condition(
        patient_id      = pat_id,
        block_name      = "LM",
        onsets          = on_abs,
        offsets         = off_abs,
        sampling_rate   = fs,
        trial_ids       = trial_ids,
        out_dir         = str(prep_dir),
        condition_name  = condition_name,
        resp_accuracy   = resp_accuracy,
        trial_idx       = trial_idx_col,
        trial_end_abs   = _trial_end_abs,
    )
    return prep_dir, fs


# -----------------------------------------------------------------------------
# Build combined (macro + micro) signal matrix
# -----------------------------------------------------------------------------
def build_combined_signals(data_ecog, data_micro, chans_ecog, chans_micro):
    """
    Stack macros and micros into one (n_samples, n_macro + n_micro) matrix.
    Returns (signals, names, is_micro_mask).
    """
    signals  = np.concatenate([data_ecog, data_micro], axis=1)
    names    = list(chans_ecog) + list(chans_micro)
    is_micro = np.array([False]*len(chans_ecog) + [True]*len(chans_micro), dtype=bool)
    return signals, names, is_micro


# -----------------------------------------------------------------------------
# WM rereferencing with optional application to micros
# -----------------------------------------------------------------------------
def apply_wm_reref_selective(signals, names, wm_names, is_micro, *,
                             apply_wm_to_micros=False, min_wm=3):
    """
    Apply WM rereferencing via lf_ersp.apply_wm_reference_with_exclusions.

    If apply_wm_to_micros=False (default): only macros are rereferenced; micros pass through.
    """
    from lf_ersp import apply_wm_reference_with_exclusions

    wm_set = set(wm_names)
    wm_idx = [i for i, nm in enumerate(names) if nm in wm_set]

    if apply_wm_to_micros:
        return apply_wm_reference_with_exclusions(
            signals, names, wm_idx, bad_channels_for_ref=[], min_wm=min_wm)

    # Rereference macros only
    macro_idx   = np.where(~is_micro)[0]
    macro_sig   = signals[:, macro_idx]
    macro_names = [names[i] for i in macro_idx]
    wm_in_macro = [int(np.where(macro_idx == i)[0][0]) for i in wm_idx if i in macro_idx]

    macro_rr, used, excl = apply_wm_reference_with_exclusions(
        macro_sig, macro_names, wm_in_macro, bad_channels_for_ref=[], min_wm=min_wm)

    out = signals.copy()
    out[:, macro_idx] = macro_rr
    return out, used, excl


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_macro_ersp(ersp, *, save_path, patient_id, condition, chan_name, params):
    """Thin wrapper around lf_ersp.plot_ersp for a single macro channel."""
    from lf_ersp import plot_ersp
    save_dir = os.path.dirname(save_path) or None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    return plot_ersp(ersp, patient_id=patient_id, condition=condition,
                     reref_type="WM", chan_name=chan_name,
                     save_dir=save_dir, params=params, save_sidecar=False)


def plot_tetrode_ersp_2x2(ersps, *, save_path, patient_id, condition,
                          tetrode_names, tetrode_idx, params):
    """
    2x2 subplot of 4 micro ERSPs (one tetrode).

    Parameters
    ----------
    ersps : list[dict] of length 4 (returns of lf_ersp.compute_ersp)
    tetrode_names : list[str] of length 4
    tetrode_idx : int  — 1-based index used in title/filename
    """
    from lf_ersp import _centers_to_edges

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    mode = ersps[0]["meta"]["mode"]

    for ax, ersp, name in zip(axes.ravel(), ersps, tetrode_names):
        f, x, A = ersp["f"], ersp["x"], ersp["avg_db"]
        te, fe  = _centers_to_edges(x), _centers_to_edges(f)
        im = ax.pcolormesh(te, fe, A, shading="flat", cmap="bwr",
                           vmin=params.vmin, vmax=params.vmax)
        ax.set_ylim(0, params.fmax)
        ax.set_title(name, fontsize=10)

        if mode == "TN":
            nB, nS, _ = ersp["meta"]["bins"]
            ax.axvline(te[nB],      color="k", lw=1.2)
            ax.axvline(te[nB + nS], color="k", lw=1.2)
        else:
            ax.axvline(0.0, color="k", lw=1.2)
            off = ersp["markers"].get("offset")
            if off is not None:
                ax.axvline(off, color="k", ls="--", lw=1.2)

    xlabel = "Norm. time (%)" if mode == "TN" else "Time (s)"
    for ax in axes[-1, :]: ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:  ax.set_ylabel("Frequency (Hz)")

    fig.suptitle(f"{patient_id} – {condition} – WM-ref ERSP: tetrode {tetrode_idx}", fontsize=11)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="dB")

    save_dir = os.path.dirname(save_path) or None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=300, format="tiff", transparent=True, bbox_inches="tight")
    plt.close(fig)
    return save_path
