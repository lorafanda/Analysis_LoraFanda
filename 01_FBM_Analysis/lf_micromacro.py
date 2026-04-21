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
def _read_channel_struct_names(f, struct_key):
    """Extract the 'name' field from a MATLAB struct array stored in HDF5."""
    name_refs = f[struct_key]["name"][()].ravel()
    return ["".join(chr(int(c)) for c in f[r][()].ravel() if int(c) != 0)
            for r in name_refs]


def load_microepi_mat(path):
    """
    Load one MicroEPI export .mat (v7.3 / HDF5).

    Returns
    -------
    dict with:
        'data_ecog'   : (n_samples, n_macro) float32
        'data_micro'  : (n_samples, n_micro) float32
        'chans_ecog'  : list[str]
        'chans_micro' : list[str]
        'photodiode'  : (n_samples,) float32
        'fs'          : 2048.0
    """
    with h5py.File(path, "r") as f:
        data_ecog  = np.asarray(f["dataEcog"][()],      dtype=np.float32)
        data_micro = np.asarray(f["dataMicroDown"][()], dtype=np.float32)
        photodiode = np.asarray(f["photodiode"][()],    dtype=np.float32).ravel()

        # Normalize to (samples, channels)
        if data_ecog.shape[0]  < data_ecog.shape[1]:  data_ecog  = data_ecog.T
        if data_micro.shape[0] < data_micro.shape[1]: data_micro = data_micro.T

        chans_ecog  = _read_channel_struct_names(f, "chansEcog")
        chans_micro = _read_channel_struct_names(f, "chansMicro")

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
# WM channel derivation from BIDS electrodes TSV
# -----------------------------------------------------------------------------
def derive_wm_channels_from_electrodes_tsv(tsv_path_pattern):
    """
    Derive WM channel names from a BIDS electrodes TSV.

    Rule: tissueWeights_1 == 1 AND first token of tissueLabel starts with 'wm-'.
    """
    matches = glob.glob(tsv_path_pattern)
    if not matches:
        raise FileNotFoundError(f"No electrodes TSV matching: {tsv_path_pattern}")
    df = pd.read_csv(matches[0], sep="\t")

    def _is_wm(row):
        try:
            w1 = float(row.get("tissueWeights_1", np.nan))
        except (TypeError, ValueError):
            return False
        tokens = str(row.get("tissueLabel", "")).strip().split()
        return np.isclose(w1, 1.0) and bool(tokens) and tokens[0].startswith("wm-")

    return [str(r["name"]) for r in df.to_dict("records") if _is_wm(r)]


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
def extract_events_from_photodiode(photodiode, fs, *, trig_name="photodiode", **pd_kwargs):
    """
    Run square-wave photodiode detection from a 1-D photodiode trace.
    Wraps LFfunctions_PDextract.get_trigger_indexes_photodiode.
    """
    from LFfunctions_PDextract import get_trigger_indexes_photodiode

    pd_2d = np.asarray(photodiode, dtype=np.float32).reshape(-1, 1)
    out = get_trigger_indexes_photodiode(
        raw_signals=pd_2d, sampling_rate=float(fs),
        channel_names=[trig_name], trig_name=trig_name,
        time_range=(0, -1), do_plot=False, **pd_kwargs,
    )
    on_abs, off_abs = out[0], out[1]
    return np.asarray(on_abs, dtype=np.int64), np.asarray(off_abs, dtype=np.int64)


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
