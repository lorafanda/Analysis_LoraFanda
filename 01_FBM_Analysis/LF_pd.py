# LF_pd.py
# Unified raw loader for PAT (TRC), EL Bern (EDF/H5), and MicroEpi per-electrode .mat folders.
# Returns a consistent (raw_signals [samples×channels], sampling_rate [Hz], channel_names),
# plus base_path, save_path, patient_id, and discovered onset files.
#
# Usage:
#   from LF_pd import load_patient_raw
#   info = load_patient_raw(patient_ids[idx], block_name='LM', verbose=True)
#   raw_signals = info['raw_signals']
#   sampling_rate = info['sampling_rate']
#   channel_names = info['channel_names']
#   base_path = info['base_path']
#   save_path = info['save_path']
#   patient_id = info['patient_id']
#   matching_files_onsets = info['matching_files_onsets']

from __future__ import annotations

import os
import re
import glob
import numpy as np
from typing import Dict, Tuple, List, Any, Optional

# Optional imports are inside functions to avoid hard deps unless the path is used.

__all__ = ["load_patient_raw"]


# ----------------------------- PATH HELPERS -----------------------------

def _paths_for_patient(patient_item: Any, block_name: str) -> Tuple[str, str, str]:
    """
    Decide patient_id, base_path, save_path based on cohort:
      - Integers -> PAT_{id} (HUG) TRC path
      - Strings starting with 'EL' -> Bern EL cohort (EDF/H5)
      - Other strings -> MicroEPI-G-xx cohort (per-electrode .mat)
    """
    if isinstance(patient_item, str) and patient_item.upper().startswith("EL"):
        patient_id = patient_item  # e.g., 'EL033'
        base_path = rf"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_BERN\{patient_id}\task_FBM\data_{block_name}\raw"
        save_path = os.path.dirname(base_path)   # ".../task_FBM/data_LM" — parse_and_save appends "prep0"
        return patient_id, base_path, save_path
    
    if isinstance(patient_item, str):  # MicroEpi 'G-01', 'G-02', ...
        patient_id = f"MicroEPI-{patient_item}"
        base_path = rf"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\MICROEPI\{patient_id}\task_FBM\data_{block_name}\raw"
        save_path = os.path.join(os.path.dirname(base_path), "prep0")
        return patient_id, base_path, save_path

    # Default: HUG SEEG PAT_xxxx (TRC)
    patient_id = f"PAT_{int(patient_item)}"
    base_path = rf"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_HUG\{patient_id}\task_FBM\data_{block_name}\raw"
    save_path = os.path.join(os.path.dirname(base_path), "prep0")
    return patient_id, base_path, save_path


def _find_onset_files(base_path: str) -> List[str]:
    """Find onset/annotation files (.txt preferred else .tsv) in base_path."""
    txts = glob.glob(os.path.join(base_path, "*.txt"))
    if txts:
        return txts
    return glob.glob(os.path.join(base_path, "*.tsv"))


# ----------------------------- UTILITIES -----------------------------

_NSRE = re.compile(r"([0-9]+)")

def _natural_key(s: Any) -> List[Any]:
    return [int(t) if str(t).isdigit() else str(t).lower() for t in _NSRE.split(str(s))]

def _to_str_list(arr: Any) -> List[str]:
    """Robustly convert MATLAB-ish label arrays (bytes, char arrays, objects) to List[str]."""
    out: List[str] = []
    flat = np.ravel(arr)
    for v in flat:
        if isinstance(v, (bytes, np.bytes_)):
            out.append(v.decode("utf-8", errors="ignore"))
        elif isinstance(v, np.ndarray) and v.dtype.kind in ("S", "U", "i"):
            # char array or numeric chars -> join
            try:
                out.append("".join(chr(int(c)) for c in v.flatten()))
            except Exception:
                out.append(str(v))
        else:
            out.append(str(v))
    return out


# ----------------------------- LOADERS -----------------------------

def _load_trc(trc_path: str) -> Tuple[np.ndarray, float, np.ndarray]:
    """Micromed TRC via Neo → (samples×channels, fs, names)."""
    import neo
    block = neo.io.MicromedIO(filename=trc_path).read_block()
    analog_signal = block.segments[0].analogsignals[0]
    channel_names = np.array(list(analog_signal.array_annotations["channel_names"]), dtype=object)
    fs = float(analog_signal.sampling_rate)
    X = np.asarray(analog_signal).astype(np.float32, copy=False)  # [samples, channels]
    return X, fs, channel_names


def load_edf(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw_signals = raw.get_data().T
    sampling_rate = float(raw.info["sfreq"])
    channel_names = list(map(str, raw.ch_names))
    return raw_signals, sampling_rate, channel_names

def _load_el_edf(edf_path: str) -> Tuple[np.ndarray, float, np.ndarray]:
    """EL cohort EDF via Neo → (samples×channels, fs, names)."""
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw_signals = raw.get_data().T
    sampling_rate = float(raw.info["sfreq"])
    channel_names = list(map(str, raw.ch_names))
    return raw_signals, sampling_rate, channel_names
    

# --- REPLACE the _load_el_h5 in LF_pd.py with this version ---

def _load_el_h5(h5_path: str) -> Tuple[np.ndarray, float, np.ndarray]:
    """EL cohort H5 → (samples×channels, fs, names). Supports '/traces/raw' layout."""
    import h5py, numpy as np

    with h5py.File(h5_path, "r") as f:
        # --- Preferred case: channels as 1D datasets under /traces/raw ---
        if "/traces/raw" in f and isinstance(f["/traces/raw"], h5py.Group):
            grp = f["/traces/raw"]

            # fs from group attrs (your file has sfreq=1024.0 here)
            fs = None
            for k in ("sfreq", "fs", "srate", "sampling_rate", "Fs", "SRate", "SR"):
                if k in grp.attrs:
                    try:
                        fs = float(np.asarray(grp.attrs[k]).squeeze())
                        break
                    except Exception:
                        pass
            if fs is None:
                # very light fallback: try /meta attrs
                if "/meta" in f and isinstance(f["/meta"], h5py.Group):
                    for k in ("sfreq", "fs", "srate", "sampling_rate"):
                        if k in f["/meta"].attrs:
                            try:
                                fs = float(np.asarray(f["/meta"].attrs[k]).squeeze())
                                break
                            except Exception:
                                pass
            if fs is None:
                raise RuntimeError("H5 loader: sampling rate not found (expected 'sfreq' in /traces/raw attrs).")

            # collect all 1D numeric datasets (channels)
            ds_names = []
            for name, obj in grp.items():
                if isinstance(obj, h5py.Dataset) and obj.ndim == 1 and np.issubdtype(obj.dtype, np.number):
                    ds_names.append(name)
            if not ds_names:
                raise RuntimeError("H5 loader: /traces/raw has no 1D numeric datasets.")

            # pick the most common length (align channels)
            from collections import Counter
            lengths = Counter(grp[n].shape[0] for n in ds_names)
            L_common = lengths.most_common(1)[0][0]
            keep = [n for n in ds_names if grp[n].shape[0] == L_common]

            # natural-sort channel names like DC5, DC6, A_L1... (uses _natural_key defined above)
            keep = sorted(keep, key=_natural_key)

            X = np.stack([grp[n][()] for n in keep], axis=1).astype(np.float32, copy=False)  # [samples × channels]
            names = np.array(keep, dtype=object)
            return X, float(fs), names

        # --- Generic fallback: any 2D numeric dataset anywhere ---
        ds2d = []
        def visit(n, o):
            if isinstance(o, h5py.Dataset) and o.ndim == 2 and np.issubdtype(o.dtype, np.number):
                ds2d.append((n, o.shape))
        f.visititems(visit)

        if not ds2d:
            raise RuntimeError("H5 loader: no usable datasets (expected /traces/raw or a 2D numeric dataset).")

        # pick the largest 2D dataset by area
        name, shape = max(ds2d, key=lambda x: x[1][0] * x[1][1])
        X = np.asarray(f[name])
        if X.shape[0] < X.shape[1]:
            X = X.T  # [samples × channels]

        # fs (fallback search at file level)
        fs = None
        for key in ("sfreq", "fs", "srate", "sampling_rate", "Fs", "SRate", "SR"):
            if key in f.attrs:
                try:
                    fs = float(np.asarray(f.attrs[key]).squeeze()); break
                except Exception:
                    pass
        if fs is None:
            raise RuntimeError("H5 loader: sampling rate not found in file attrs for 2D fallback.")

        # names (generic)
        names = np.array([f"ch{i+1}" for i in range(X.shape[1])], dtype=object)
        return X.astype(np.float32, copy=False), float(fs), names




def _load_microepi_folder(folder: str) -> Dict[str, Tuple[np.ndarray, float, np.ndarray]]:
    """
    Load all per-electrode *_micro.mat / *_macro.mat files → dict:
      {'micro': (X, fs, names), 'macro': (...)}
    """
    from scipy.io import loadmat
    out: Dict[str, Tuple[np.ndarray, float, np.ndarray]] = {}
    for kind in ("micro", "macro"):
        files = sorted(glob.glob(os.path.join(folder, f"*_{kind}.mat")), key=_natural_key)
        if not files:
            continue
        signals: List[np.ndarray] = []
        ch_names: List[str] = []
        fs_all: List[float] = []
        nmin: Optional[int] = None
        for f in files:
            md = loadmat(f, squeeze_me=True, struct_as_record=False)
            if "data" not in md or "sampling_rate" not in md:
                raise RuntimeError(f"MicroEpi loader: missing 'data'/'sampling_rate' in {os.path.basename(f)}")
            x = md["data"].astype(np.int16, copy=False)
            fs = float(md["sampling_rate"])
            fs_all.append(fs)
            nmin = x.size if (nmin is None or x.size < nmin) else nmin
            # channel name from filename tail: *_<CHAN>_micro.mat
            base = os.path.basename(f)
            m = re.match(r".*_(?P<chan>[^_]+)_(?:micro|macro)\.mat$", base, re.IGNORECASE)
            ch = m.group("chan") if m else os.path.splitext(base)[0]
            ch_names.append(str(ch))
            signals.append(x)
        if len(set(fs_all)) != 1:
            raise RuntimeError(f"MicroEpi loader: inconsistent fs for {kind}: {set(fs_all)}")
        fs_val = fs_all[0]
        X = np.stack([s[:nmin] for s in signals], axis=1).astype(np.float32, copy=False)  # [samples, channels]
        order = sorted(range(len(ch_names)), key=lambda i: _natural_key(ch_names[i]))
        X = X[:, order]
        ch_names = [ch_names[i] for i in order]
        out[kind] = (X, fs_val, np.array(ch_names, dtype=object))
    return out


# ----------------------------- PUBLIC API -----------------------------

def load_patient_raw(
    patient_item: Any,
    block_name: str = "LM",
    *,
    use_el_mat_fallback: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Load raw data for a patient into a unified structure (samples×channels, fs, names),
    discovering files in this priority:
      PAT (HUG): TRC
      EL (Bern): EDF → H5 → (optional) *_LM_data_raw.mat
      MicroEpi: per-electrode *_micro.mat / *_macro.mat (prefer stream containing 'ainp1' if present)

    Parameters
    ----------
    patient_item : int | str
        e.g., 3390  → PAT_3390 (TRC),
              'EL033' → Bern EL cohort (EDF/H5),
              'G-02' → MicroEPI-G-02 (per-electrode .mat)
    block_name : str
        e.g., 'LM', 'CL', ...
    use_el_mat_fallback : bool
        If True, allow EL *_LM_data_raw.mat when EDF/H5 are absent. Default False.
    verbose : bool
        Print concise status.

    Returns
    -------
    dict with keys:
      'patient_id', 'base_path', 'save_path',
      'raw_signals' (float32 [samples×channels]),
      'sampling_rate' (float Hz),
      'channel_names' (np.array of str),
      'matching_files_onsets' (list[str])
    """
    patient_id, base_path, save_path = _paths_for_patient(patient_item, block_name)

    if verbose:
        print(f"*** {patient_id}\n{base_path}")

    # Find onset files (txt preferred else tsv)
    matching_files_onsets = _find_onset_files(base_path)
    if verbose:
        if matching_files_onsets:
            print("\n*** Found onset file(s):")
            for p in matching_files_onsets:
                print("   ", p)
        else:
            print("\n*** Found onset file(s): (none)")

    # --------- Discovery order ----------
    # PAT: TRC
    # EL: EDF → H5 → (optional .mat)
    # MicroEpi: per-electrode .mat
    raw_signals: Optional[np.ndarray] = None
    sampling_rate: Optional[float] = None
    channel_names: Optional[np.ndarray] = None

    is_el = isinstance(patient_item, str) and patient_item.upper().startswith("EL")

    # 1) TRC (present mostly for PAT/HUG)
    trcs = glob.glob(os.path.join(base_path, "*.TRC"))
    if trcs:
        if verbose:
            print("\n*** Found TRC:")
            print("   ", trcs[0])
        raw_signals, sampling_rate, channel_names = _load_trc(trcs[0])

    # 2) EL cohort: EDF → H5 (skip EL .mat unless allowed)
    if raw_signals is None and is_el:
        edfs = glob.glob(os.path.join(base_path, f"*_{block_name}.edf"))
        if edfs:
            if verbose:
                print("\n*** Found EL EDF:")
                print("   ", edfs[0])
            raw_signals, sampling_rate, channel_names = _load_el_edf(edfs[0])

    if raw_signals is None and is_el:
        h5s = glob.glob(os.path.join(base_path, f"*_{block_name}.h5")) + \
              glob.glob(os.path.join(base_path, f"*_{block_name}.hdf5"))
        if h5s:
            if verbose:
                print("\n*** Found EL H5:")
                print("   ", h5s[0])
            raw_signals, sampling_rate, channel_names = _load_el_h5(h5s[0])

    if raw_signals is None and is_el and use_el_mat_fallback:
        el_mats = glob.glob(os.path.join(base_path, f"*_{block_name}_data_raw.mat"))
        if el_mats:
            if verbose:
                print("\n*** Found EL raw .mat (fallback):")
                print("   ", el_mats[0])
            # Minimal EL .mat loader (expects keys: dataEcogUp/labelsEcog/srate or data/channel_names/sampling_rate)
            from scipy.io import loadmat as _lm
            md = _lm(el_mats[0], squeeze_me=True, struct_as_record=False)
            data = md.get("dataEcogUp", md.get("data", None))
            labels = md.get("labelsEcog", md.get("channel_names", None))
            fs = md.get("srate", md.get("sampling_rate", None))
            if data is None or labels is None or fs is None:
                raise RuntimeError(f"EL .mat fallback: missing one of data/labels/fs in {os.path.basename(el_mats[0])}")
            X = np.asarray(data)
            if X.ndim != 2:
                raise RuntimeError(f"EL .mat fallback: expected 2D, got {X.shape}")
            names = np.array(_to_str_list(labels), dtype=object)
            if X.shape[1] != len(names) and X.shape[0] == len(names):
                X = X.T
            if X.shape[1] != len(names):
                raise RuntimeError(f"EL .mat fallback: names {len(names)} != channels {X.shape[1]}")
            raw_signals = X.astype(np.float32, copy=False)
            sampling_rate = float(np.asarray(fs).squeeze())
            channel_names = names

    # 3) MicroEpi per-electrode .mat
    if raw_signals is None:
        buckets = _load_microepi_folder(base_path)
        if not buckets:
            raise RuntimeError("No compatible raw found: TRC/EDF/H5/EL(.mat)/MicroEpi per-electrode.")
        # prefer stream containing 'ainp1'
        picked_kind: Optional[str] = None
        for kind in ("micro", "macro"):
            if kind in buckets:
                X, fs_val, chs = buckets[kind]
                if any(str(c).lower() == "ainp1" for c in chs):
                    raw_signals, sampling_rate, channel_names = X, fs_val, chs
                    picked_kind = kind
                    break
        if raw_signals is None:
            # fallback to 'micro' else first available
            if "micro" in buckets:
                raw_signals, sampling_rate, channel_names = buckets["micro"]
                picked_kind = "micro"
            else:
                picked_kind = next(iter(buckets.keys()))
                raw_signals, sampling_rate, channel_names = buckets[picked_kind]
        if verbose:
            print(f"\n*** Loaded MicroEpi {picked_kind}: {raw_signals.shape[0]} samples × {raw_signals.shape[1]} channels @ {sampling_rate} Hz")

    # Final sanity
    if raw_signals is None or sampling_rate is None or channel_names is None:
        raise RuntimeError("Loader internal error: incomplete outputs.")

    if verbose:
        print(f"*** Loaded: {raw_signals.shape[0]} samples × {raw_signals.shape[1]} channels @ {sampling_rate} Hz")
        preview = ", ".join(list(map(str, channel_names[:24])))
        print(f"*** First 24 channels: {preview}")

    return {
        "patient_id": patient_id,
        "base_path": base_path,
        "save_path": save_path,
        "raw_signals": raw_signals,           # float32 [samples × channels]
        "sampling_rate": float(sampling_rate),
        "channel_names": np.array(channel_names, dtype=object),
        "matching_files_onsets": matching_files_onsets,
    }
