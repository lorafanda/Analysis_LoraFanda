# lf_io_utils.py
from __future__ import annotations
import os, json, h5py
import numpy as np
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

# ---- AUX removal (bug-fixed) ----
AUX_DROP_PREFIXES = ("MRK", "MKR", "X", "ECG", "PHOTO", "AUDIO")
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

# ---- WM reference maps (unchanged data, compact helpers) ----
WHITE_MATTER_REFS_NUMS = {
    "PAT_3066": {"FIG":[6,5], "FOG":[15], "FOD":[12], "CAG":[11,10,9,8,7], "CAD":[8,6,5,3,2],
                 "IAG":[18,11,9,3,2], "IMG":[18,9,8,7,6,3], "AG":[3,2,1], "HAG":[1], "HPG":[7,2],
                 "PPG":[6,5,4,2], "TIG":[3], "AD":[8,6,4,3,2,1], "HAD":[3,2,1], "HPD":[2,1], "PPD":[6,5,4,3]},
    "PAT_3975": {"FOG":[15,10,8,3,1], "CPG":[10,6,5,4,3], "AG":[12], "HAG":[5,2,1], "PHG":[6,4],
                 "IAG":[8,6,4,3,2,1], "IMG":[13,12,11,6,3], "FOD":[14,13,12,11,9,6,5,3],
                 "CPD":[15,8,6,2], "AD":[7,4,3,2,1], "HAD":[3,2,1], "PHD":[11,7], "IAD":[10],
                 "IMD":[17,16,15,14,12,10,9,7]},
    "PAT_3415": {"IMG":[16,15,12,11,8,6,5,4,3,2], "IPG":[12,10,8,3,2,1], "HLG":[17,16,3,2]},
    "PAT_3965": {"ag":[3,4,5,6], "hpg":[8,9,10,11], "imd":[6,7,8,9,10,11,12]},
    "PAT_2868": {"POP":[3], "IDM":[4,1], "SMA":[2,1], "PPS":[3,2,1], "PRI":[3,2,1], "POM":[4,1],
                 "PPI":[2], "POI":[3,2,1]},
    "PAT_3455": {"AD":[2,1], "HAD":[2,1], "TOD":[6], "CPD":[12,11,10,9,6,5,2], "OPD":[5,4,3,2],
                 "PHD":[5], "OTD":[8], "TSP":[8,6,5,4], "IMD":[11,5,4,3], "IPD":[14,11]},
    "PAT_3390": {"IAG":[18,17,16,9,1], "CAG":[8,6,5,4,2], "CPG":[15,13,10,9,1], "HPG":[10,9,8],
                 "AG":[12,11], "HAG":[12,11,10,9], "FOG":[9,4,3,2,1]},
    "MicroEPI-B-01": {"pI_L":[8,9,10,11,12,13,14], "A_L":[4,5],"ITG_L":[6,9],"sSMG_L":[2,3,5,6],"IOG_L":[4,5,7],"LinG_L":[4,5]},
    "EL042": {"OF_R":[10,11],"aI_R":[1,2,3,4,5,6,7,10,11,12,13,14,17],"A_R":[5,6,7],"aH_R":[7],"pH_R":[7,8],"STG_R":[3],
             "A_L":[6,7,8,9]},
    "EL040":{"FP_R":[8],"OF_R":[4,5,9,10,11,12,13,14,15],"CinG_R":[5,6,7,8,9,10,11,12],"aI_R":[9,10,11,12,13,14,15],
             "A_R":[8,9],"EntG_R":[5,6,8],"aH_R":[8],"PHG_R":[5,6,7],"pH_R":[4,5,6,7,8,9],"STG_R":[1,8],"iNH_R":[13,14,15],
             "CinG_L":[5,6,7,8,9,10],"pPVH_L":[7,8,9]},
    "EL039":{"OF_R":[10,11,12,13],"pSTG_R":[2],"CinG_R":[2,3,4,5,6],"pH_L":[6,7],"LinG_R":[9],
             "STO_R":[6,7,8,9],"aSTG_R":[2],"A_L":[4,5,6]},
    "EL038":{"OF_R":[10,11,12],"CinG_R":[2,3,4,5,6],"aSTG_R":[2],"pSTG_R":[2],"STO_R":[6,7,8,9],"A_L":[4,5,6],"pH_L":[6,7]},
    "EL037":{"A_L":[5],"EntG_L":[7],"pH_L":[6],"CinG_L":[3,4,5,6,7,8,9,10,11],"aI_L":[12,13,14,15],"pI_L":[11,12],"A_R":[7,8]},
    "EL036":{"A_R":[4,5,6,7],"aH_R":[5,6],"mH_R":[10],"PfC_R":[5]},
    "EL035":{"A_L":[8],"aH_L":[7],"A_R":[6,7,8],"aH_R":[7,8],"pH_R":[4,5,6,7],"ParaHG_R":[7,8,9],"OF_R":[4,5,6,7,8,9,10],"CinG_R":[4,5,6],"aI_R":[8,9,10,11],"pI_R":[11,12,13]},
    "EL034":{"CinG_R":[4,5,6,8,9],"OFG_L":[4,5],"pSFG_L":[4],"aH_L":[6,7],"MFG_L":[3]},
    "EL030":{"A-L":[6,7,8],"aH-L":[6,7],"A-R":[7,8],"aH-R":[6,7],"pH-R":[3,4,5]},
    # "MicroEPI": {"ag":[3,4,5,6], "hpg":[8,9,10,11], "imd":[6,7,8,9,10,11,12]},

    # "PAT_3965": {"CAD": [7],"CAG": [8, 2],"CMD": [12, 7, 6, 3],"CMG": [9, 8, 7, 6, 4],"CPD": [15, 14, 5],
    #              "CPG": [10, 9],"HAD": [7],"HAG": [7],"HPD": [5],"HPG": [12, 5],"IAD": [16, 15, 10, 9, 8, 7, 6, 5, 1],
    #              "IAG": [13, 10, 7],"IMD": [4],"IMG": [18, 16, 13, 12, 11, 9, 8, 3],"POD": [1],"POG": [3]},
}
def _expand_labels(prefix: str, nums): return [f"{prefix}{int(n)}" for n in nums]
WHITE_MATTER_REFS_LABELS = {pid:{lead:_expand_labels(lead, nums) for lead,nums in leads.items()}
                            for pid, leads in WHITE_MATTER_REFS_NUMS.items()}
import re


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

# Replace the two WM helpers with normalized versions
def wm_labels_for_patient(patient_id: str):
    leads = WHITE_MATTER_REFS_LABELS.get(patient_id, {})
    # normalize the expanded labels
    labs = []
    for _, lablist in leads.items():
        labs.extend([normalize_label(l) for l in lablist])
    return set(labs)

def wm_indices_for_patient(patient_id: str, channel_names):
    want = wm_labels_for_patient(patient_id)
    # compare on normalized names
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
