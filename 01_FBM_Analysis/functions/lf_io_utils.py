# lf_io_utils.py
from __future__ import annotations
import os, json, glob, h5py
import numpy as np
import pandas as pd
import re
import mne

try:
    import mne
except Exception:
    mne = None
try:
    import neo
except Exception:
    neo = None

# ---- Verbose logging ----
VERBOSE = True
def set_verbose(flag: bool = True):
    global VERBOSE; VERBOSE = bool(flag)
def log(msg: str):
    if VERBOSE:
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[LF {ts}] {msg}")

# ---- AUX removal ----
AUX_DROP_PREFIXES = ("MRK", "MKR", "X", "ECG", "EX", "AUDIO")
# AUX_DROP_PREFIXES = ("MRK", "MKR", "X", "ECG", "EX", "AUDIO","POP","IDM","IDP","SMA","PRS","PRI","POS","POM","PPI","POP","PPS")

def _is_aux_channel(name: str) -> bool:
    if not name:
        return False
    s = str(name).strip().upper()
    if s.endswith(("+", "-")):
        s = s[:-1]
    return any(s.startswith(p) for p in AUX_DROP_PREFIXES)

def filter_aux_channels(signals: np.ndarray, channel_names: list[str]):
    if signals is None or channel_names is None:
        raise ValueError("signals and channel_names must be provided")
    if signals.ndim != 2:
        raise ValueError("signals must be 2D (n_samples, n_channels)")

    # enforce (n_samples, n_channels)
    n0, n1 = signals.shape
    if n0 < n1:  # most TRC/EDF loaders return (n_samples, n_channels); keep it consistent
        pass
    n_samples, n_channels = signals.shape
    if len(channel_names) != n_channels:
        raise ValueError(f"{len(channel_names)} names vs {n_channels} channels in signals")

    kept_idx, dropped_idx = [], []
    for i, nm in enumerate(channel_names):
        if _is_aux_channel(nm):
            dropped_idx.append(i)
        else:
            kept_idx.append(i)
    filt_signals = signals[:, kept_idx]
    filt_names   = [channel_names[i] for i in kept_idx]
    dropped_names= [channel_names[i] for i in dropped_idx]
    return filt_signals, filt_names, kept_idx, dropped_idx, dropped_names

def drop_aux_from_names(channel_names):
    kept_idx, dropped_idx = [], []
    for i, nm in enumerate(channel_names):
        if _is_aux_channel(nm):
            dropped_idx.append(i)
        else:
            kept_idx.append(i)
    filt_names    = [channel_names[i] for i in kept_idx]
    dropped_names = [channel_names[i] for i in dropped_idx]
    return filt_names, kept_idx, dropped_idx, dropped_names

# ---- WM reference: derive from BIDS electrodes TSV (no hardcoded lists) ----
#
# WM channels are read from `*_electrodes.tsv` per patient. Rule:
#     tissueWeights_1 == 1.0  AND  first token of tissueLabel starts with "wm-"
#
# Path patterns (UNC) per cohort:
#     EL:        ...\DATARAW\SEEG_EXPERIMENTS_BERN\Reconstruction\<pid>\BIDS\ieeg\sub-<pid>_electrodes.tsv
#     PAT:       ...\#SHARE\To_send_collaborators\PAT_<num>\BIDS\ieeg\sub-<num>_electrodes.tsv
#     MicroEPI:  ...\DATARAW\BIDS_elec\MICROEPI\sub-<full_id>\ieeg\*_electrodes.tsv
#
# `electrodes_tsv_path_for_patient` returns a glob pattern; the helper
# resolves it against the filesystem.

NASAC_ROOT = r"\\nasac-m2.unige.ch\m-HumanNeuronLab"

def electrodes_tsv_path_for_patient(patient_id) -> str:
    """Return a BIDS electrodes-TSV path pattern for the given patient id."""
    pid = str(patient_id).strip()
    if pid.startswith(("MicroEPI", "G-", "B-")):
        sub = pid if pid.startswith("MicroEPI") else f"MicroEPI-{pid}"
        return fr"{NASAC_ROOT}\DATARAW\BIDS_elec\MICROEPI\sub-{sub}\ieeg\*_electrodes.tsv"
    if pid.startswith("EL"):
        return fr"{NASAC_ROOT}\DATARAW\BIDS_elec\SEEG-BERN\sub-{pid}\ieeg\*_electrodes.tsv"
    num = pid.replace("PAT_", "")
    return fr"{NASAC_ROOT}\DATARAW\BIDS_elec\SEEG-HUG\sub-{num}\ieeg\*_electrodes.tsv"


def derive_wm_channels_from_electrodes_tsv(tsv_path_pattern: str) -> list[str]:
    """
    Read a BIDS electrodes TSV and return the names of channels flagged as WM.

    Rule: `tissueWeights_1 == 1.0` and first token of `tissueLabel` starts with `wm-`.
    Accepts a literal path or a glob pattern; the first match is used.
    """
    matches = glob.glob(tsv_path_pattern)
    if not matches:
        raise FileNotFoundError(f"No electrodes TSV matching: {tsv_path_pattern}")
    df = pd.read_csv(matches[0], sep="\t")

    def _is_wm(row) -> bool:
        try:
            w1 = float(row.get("tissueWeights_1", np.nan))
        except (TypeError, ValueError):
            return False
        tokens = str(row.get("tissueLabel", "")).strip().split()
        return (w1 > 0.97) and bool(tokens) and tokens[0].startswith("wm-")
    return [str(r["name"]) for r in df.to_dict("records") if _is_wm(r)]


def canonicalize_labels(labels: list[str], patient_id_hint: str | None = None) -> tuple[list[str], dict]:
    """
    Return (canon_list, mapping) where mapping[orig] = canon.
    """
    mapping = {lab: canonicalize_label(lab, patient_id_hint) for lab in labels}
    canon_list = [mapping[lab] for lab in labels]
    return canon_list, mapping

def canonicalize_label(label: str, patient_id_hint: str | None = None) -> str:
    """
    Comparable name across sites:
      - Strip '<PAT_####_...>' or '<EL###_...>' prefix if present (when patient_id_hint is given)
      - Drop trailing + / - polarity
      - Remove underscores/hyphens entirely
      - Uppercase
    """
    if not label: return ""
    s = str(label).strip()
    s = re.sub(r'[+-]$', '', s)  # drop trailing polarity

    if patient_id_hint:
        pid = str(patient_id_hint).strip()
        pid_core = pid.replace("PAT_", "").replace("MicroEPI-", "")
        s = re.sub(rf'^(?:{re.escape(pid)}|{re.escape(pid_core)})_', '', s, flags=re.IGNORECASE)

    s = s.replace('_', '').replace('-', '')
    return s.upper()

def normalize_label(s: str) -> str:
    if s is None: return ""
    return str(s).replace('_', '').replace('-', '').upper()


def normalize_names(names: list[str]) -> list[str]:
    return [normalize_label(nm) for nm in names]

def wm_labels_for_patient(patient_id: str, *,
                          electrodes_tsv_pattern: str | None = None) -> set[str]:
    """
    Return WM channel names for a patient, derived from the BIDS electrodes TSV.

    Names are normalized (case-folded, separators stripped) so that downstream
    matching against raw channel labels is robust to small spelling variants.
    If `electrodes_tsv_pattern` is given, it overrides the auto-resolved path
    (used for MicroEPI .mat patients whose TSV lives outside the standard
    cohort layout). Returns an empty set if the TSV cannot be located.
    """
    pattern = electrodes_tsv_pattern or electrodes_tsv_path_for_patient(patient_id)
    try:
        names = derive_wm_channels_from_electrodes_tsv(pattern)
    except FileNotFoundError as e:
        log(f"[WM] {e}")
        return set()
    return {normalize_label(n) for n in names}


def wm_indices_for_patient(patient_id: str, channel_names, *,
                           electrodes_tsv_pattern: str | None = None) -> list[int]:
    """Return indices of WM channels in `channel_names` for the given patient."""
    want = wm_labels_for_patient(patient_id, electrodes_tsv_pattern=electrodes_tsv_pattern)
    if not want:
        return []
    lookup = {normalize_label(nm): i for i, nm in enumerate(channel_names)}
    return sorted([lookup[l] for l in want if l in lookup])


# ---- "Unknown" parcellation: drop electrodes that have no anatomical label ----
#
# Rule (mirrors the WM derivation but on the negated side):
#   An electrode is considered "Unknown" if its `tissueLabel` is empty/NaN OR
#   starts with "Unknown"/"unknown" (case-insensitive) once the leading whitespace
#   is stripped. WM channels are NOT counted as Unknown (they have a wm-* label).
#
# This is conservative: ambiguous parcellations (e.g. `ctx-lh-bankssts` with
# low tissueWeights_1) stay in. Only truly unlabelled or explicitly-Unknown
# electrodes are dropped — same source TSV as the WM derivation, so the two
# rules don't need to disagree.

def derive_unknown_channels_from_electrodes_tsv(tsv_path_pattern: str) -> list[str]:
    """
    Read a BIDS electrodes TSV and return the names of channels whose `tissueLabel`
    is empty / NaN / starts with 'Unknown' (case-insensitive). Used to drop
    unparcellated electrodes before ERSP compute so they never enter clustering.
    """
    matches = glob.glob(tsv_path_pattern)
    if not matches:
        raise FileNotFoundError(f"No electrodes TSV matching: {tsv_path_pattern}")
    df = pd.read_csv(matches[0], sep="\t")

    def _is_unknown(row) -> bool:
        raw = row.get("tissueLabel", "")
        if raw is None:
            return True
        s = str(raw).strip()
        if not s or s.lower() in ("nan", "none"):
            return True
        return s.lower().startswith("unknown")

    return [str(r["name"]) for r in df.to_dict("records") if _is_unknown(r)]


def unknown_labels_for_patient(patient_id: str, *,
                               electrodes_tsv_pattern: str | None = None) -> set[str]:
    """
    Return normalized names of unparcellated ("Unknown") electrodes for a patient.
    Returns an empty set if the electrodes TSV cannot be located — silently
    keeping all channels rather than blocking the pipeline. Pair with the
    debug report at end of 140 to spot patients where this happened.
    """
    pattern = electrodes_tsv_pattern or electrodes_tsv_path_for_patient(patient_id)
    try:
        names = derive_unknown_channels_from_electrodes_tsv(pattern)
    except FileNotFoundError as e:
        log(f"[Unknown] {e}")
        return set()
    return {normalize_label(n) for n in names}


def unknown_indices_for_patient(patient_id: str, channel_names, *,
                                electrodes_tsv_pattern: str | None = None) -> list[int]:
    """Return indices of unparcellated channels in `channel_names` for the patient."""
    want = unknown_labels_for_patient(patient_id, electrodes_tsv_pattern=electrodes_tsv_pattern)
    if not want:
        return []
    lookup = {normalize_label(nm): i for i, nm in enumerate(channel_names)}
    return sorted([lookup[l] for l in want if l in lookup])


# ---- Loaders (normalize to (n_samples, n_channels)) ----
def load_ns6(file_path):
    if neo is None:
        raise ImportError("neo is required for .ns6")
    reader = neo.rawio.BlackrockRawIO(filename=file_path)
    reader.parse_header()
    fs = float(reader.get_signal_sampling_rate())
    chunk = reader.get_analogsignal_chunk(block_index=0, seg_index=0)
    names = [n.decode() if hasattr(n, "decode") else str(n)
             for n in reader.header["signal_channels"]["name"]]
    sig = np.array(chunk)
    if sig.shape[0] < sig.shape[1]:
        sig = sig.T
    return sig, fs, names

def load_edf(file_path):
    if mne is None:
        raise ImportError("mne is required for .edf")
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    sig = raw.get_data().T  # (n_samples, n_channels)
    fs = float(raw.info["sfreq"])
    names = list(map(str, raw.ch_names))
    return sig, fs, names

def load_h5(file_path):
    with h5py.File(file_path, "r") as f:
        # --- Try your new Bern schema first: traces/raw/<channel>
        if "traces" in f and "raw" in f["traces"]:
            grp = f["traces/raw"]
            chs = list(grp.keys())
            X   = np.array([grp[ch][()] for ch in chs])  # (n_ch, n_samp)
            fs  = float(grp.attrs.get("sfreq", f.attrs.get("sfreq", np.nan)))
            names = chs
        # --- Fallback to old schema with /data + channel_names
        elif "data" in f:
            X = np.array(f["data"])
            fs = float(f.attrs.get("sampling_rate", np.nan))
            if "channel_names" in f:
                raw_names = f["channel_names"]
                names = [c.decode() if hasattr(c, "dtype") and raw_names.dtype.kind in ("S","O") else str(c)
                         for c in raw_names[()]] if hasattr(raw_names, "__getitem__") else list(raw_names)
            else:
                names = [f"Ch{i+1}"]
        else:
            raise KeyError("H5: neither 'traces/raw' nor 'data' found")

    # Normalize to (samples, channels)
    X = X.T if X.ndim == 2 and X.shape[0] < X.shape[1] else X
    if X.ndim != 2: raise ValueError("H5 'data' must be 2D")
    if X.shape[0] < X.shape[1]: X = X.T
    return X, fs, [str(n) for n in names]

def bern_good_channels_if_any(raw_dir):
    import glob, scipy.io as sio
    mats = glob.glob(os.path.join(raw_dir, "*_raw.mat"))
    if not mats: return None
    try:
        md = sio.loadmat(mats[0])
        if "good_channels" in md:
            arr = md["good_channels"]
            if arr.dtype.kind in ("U","S"):
                return [str(x).strip() for x in arr.ravel().tolist()]
            # char array (n x m)
            if arr.dtype.kind == "u" or arr.dtype.kind == "i" or arr.dtype.kind == "f":
                return None
            try:
                return ["".join(row).strip() for row in ["".join(chr(c) for c in row).strip() for row in arr]]
            except Exception:
                return None
    except Exception:
        return None


def load_first_raw_in_dir(raw_dir):
    for ext in (".TRC", ".edf", ".h5"):
        hits = [p for p in sorted(os.listdir(raw_dir)) if p.endswith(ext)]
        if hits:
            path = os.path.join(raw_dir, hits[0])
            if ext == ".TRC":  return load_trc_and_signals(path)   # (sig, names, fs)
            if ext == ".edf":  sig, fs, names = load_edf(path); return sig, names, fs
            if ext == ".h5":   sig, fs, names = load_h5(path);   return sig, names, fs
    raise FileNotFoundError("No supported raw file (.TRC/.edf/.h5) in: " + raw_dir)

    
def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    sig = np.array(data["data"])
    fs = float(data.get("sampling_rate", np.nan))
    names = [str(x) for x in data.get("channel_names", [])]
    if sig.ndim != 2:
        raise ValueError("JSON 'data' must be 2D (n_samples, n_channels)")
    if sig.shape[0] < sig.shape[1]:
        sig = sig.T
    return sig, fs, names

def load_data(file_path):
    if file_path.endswith(".ns6"): return load_ns6(file_path)
    if file_path.endswith(".edf"): return load_edf(file_path)
    if file_path.endswith(".h5"):  return load_h5(file_path)
    if file_path.endswith(".json"):return load_json(file_path)
    raise ValueError("Unsupported file format")

def load_trc_and_signals(trc_path):
    if neo is None:
        raise ImportError("neo is required for .TRC")
    block = neo.io.MicromedIO(filename=trc_path).read_block()
    ana = block.segments[0].analogsignals[0]
    names = list(ana.array_annotations["channel_names"])
    fs = float(ana.sampling_rate)
    sig = np.asarray(getattr(ana, "magnitude", ana))
    if sig.ndim != 2:
        raise ValueError(".TRC: expected 2D (samples, channels)")
    if sig.shape[0] < sig.shape[1]:
        sig = sig.T
    log(f"Loaded TRC: shape={sig.shape}, fs={fs:g} Hz")
    return sig, names, fs

# ---- Repo path helpers ----
def build_paths_for_patient(pid_raw, block_name):
    is_micro = isinstance(pid_raw, str) and pid_raw.startswith(("MicroEPI","G-","B-"))
    is_el    = isinstance(pid_raw, str) and pid_raw.startswith("EL")

    if is_micro:
        patient_id = f"MicroEPI-{pid_raw}"
        base_root  = fr"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\MICROEPI\{patient_id}\task\FBM"
    elif is_el:
        patient_id = pid_raw
        base_root  = fr"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_BERN\{patient_id}\task_FBM"
    else:
        patient_id = f"PAT_{pid_raw}"
        base_root  = fr"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_HUG\{patient_id}\task_FBM"

    raw_dir  = os.path.join(base_root, f"data_{block_name}", "raw")
    prep_dir = os.path.join(base_root, f"data_{block_name}", "prep0")
    log(f"Patient: {patient_id} | Block: {block_name}")
    log(f"Raw dir: {raw_dir}")
    log(f"Prep dir: {prep_dir}")
    return patient_id, raw_dir, prep_dir


def patient_output_dir(run_root, patient_id, block_name, product):
    path = os.path.join(run_root, patient_id, block_name, product)
    os.makedirs(path, exist_ok=True)
    return path

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

_non_neural_pat = re.compile(r"^(PHOTO|MRK|MKR|ECG|AUDIO)$|^[XE][1-8]$|^[X]$", re.I)

def _is_non_neural(label: str) -> bool:
    if not label: return False
    # Treat '-' like '_' before the heuristic checks — dash-separated names
    # (Cing-L1, OFG-L15, aH-L12, ...) are legitimate neural channels in some
    # cohorts (EL034 dykstra recon style), not bipolar markers.
    s = str(label).strip().upper().replace("-", "_")
    return ("+" in s) or bool(_non_neural_pat.match(s)) or \
           any(t in s for t in ("PHOTO", "MRK", "MKR", "ECG", "AUDIO"))