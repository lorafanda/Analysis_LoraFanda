# =========================
# Imports
# =========================
import os
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig


# =========================
# Core PD detector (monopolar)
# =========================
def _pd_detect_deriv_core(
    raw_data, *, t_start=0, t_end=None, sampling_rate=30000,
    threshold_val=0.8, flip_trigs=False, skip_samples=50, do_plot=True
):
    """
    Detect onsets from positive derivative peaks and offsets from the
    steepest negative slope after each onset. Zero-phase LP + Savitzky–Golay derivative.

    Returns
    -------
    onsets_abs, offsets_abs : np.ndarray[int]
        Absolute sample indices relative to the full recording (not window).
    """
    fs = float(sampling_rate)

    # slice
    if t_end is None or t_end < 0:
        x = np.asarray(raw_data[t_start:], float)
        base = int(t_start)
    else:
        x = np.asarray(raw_data[t_start:t_end], float)
        base = int(t_start)
    if x.size == 0:
        return np.array([], int), np.array([], int)

    if flip_trigs:
        x = -x

    # 10 Hz zero-phase low-pass (gentle smoothing)
    b, a = sig.butter(2, 10.0 / (0.5 * fs), btype='low')
    x_lp = sig.filtfilt(b, a, x)

    # Savitzky–Golay derivative (≈65.5 ms)
    sg_ms = 65.5
    win = int(round((sg_ms / 1000.0) * fs))
    win = max(win, 9)
    if win % 2 == 0: win += 1
    if win >= len(x_lp): win = max(3, ((len(x_lp) - 1) | 1))
    d = sig.savgol_filter(x_lp, window_length=win, polyorder=5, deriv=1, delta=1.0, mode='interp')
    d = d / (np.max(np.abs(d)) + 1e-12)  # max-abs normalize derivative

    # ONSETS: positive-derivative peaks
    min_dist = max(1, int(0.2 * fs))  # 0.2 s debounce
    onsets, _ = sig.find_peaks(d, height=threshold_val, distance=min_dist)

    # guard: drop too-early onset
    guard = max(skip_samples, int(0.025 * fs))
    if onsets.size and onsets[0] < guard:
        onsets = onsets[1:]

    # OFFSETS: steepest negative slope after each onset
    offsets = []
    for k, o in enumerate(onsets):
        j_start = o + max(1, int(0.02 * fs))        # 20 ms after onset
        j_stop  = onsets[k+1] if (k+1 < len(onsets)) else len(d)
        if j_stop - j_start < 3:
            continue
        f_rel = np.argmin(d[j_start:j_stop])        # most negative derivative
        f = j_start + int(f_rel)
        offsets.append(f)

    on_abs  = np.array(onsets,  int) + base
    off_abs = np.array(offsets, int) + base

    # optional quick plot (narrow)
    if do_plot:
        p1, p99 = np.percentile(x_lp, [1, 99])
        xp = np.clip((x_lp - p1) / (p99 - p1 + 1e-12), 0, 1)
        t = np.arange(len(x_lp)) / fs
        plt.figure(figsize=(18*2, 3))
        plt.plot(t, xp, alpha=0.6, label="PD (LP, norm)")
        if onsets.size:
            plt.scatter(onsets / fs, xp[np.clip(onsets, 0, len(xp)-1)],
                        marker='x', c='green', label='Onsets')
        if len(offsets):
            offsets = np.array(offsets)
            plt.scatter(offsets / fs, xp[np.clip(offsets, 0, len(xp)-1)],
                        marker='o', c='red', label='Offsets')
        plt.title("Photodiode triggers")
        plt.xlabel("Time (s)"); plt.ylabel("Norm amplitude")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

    return on_abs, off_abs


# =========================
# Label/selection helpers
# =========================
def _normalize_invalid_selector(on_abs, trial_ids, invalid_trials):
    """
    Return a boolean mask 'is_invalid' with length = len(on_abs).
    'invalid_trials' can be:
      • integer indices (0-based)
      • IDs (strings/ints) that match elements of trial_ids
    """
    n = len(on_abs)
    is_invalid = np.zeros(n, dtype=bool)
    if invalid_trials is None or n == 0:
        return is_invalid

    inv_arr = (np.asarray(list(invalid_trials))
               if not isinstance(invalid_trials, (list, tuple, set, np.ndarray))
               else np.asarray(invalid_trials))
    if inv_arr.size == 0:
        return is_invalid

    if np.issubdtype(inv_arr.dtype, np.integer):
        idx = np.clip(inv_arr.astype(int), 0, max(0, n - 1))
        is_invalid[idx] = True
        return is_invalid

    if trial_ids is not None:
        tr = np.asarray(trial_ids)
        # force trial_ids length == n (truncate/pad)
        if len(tr) > n:
            tr = tr[:n]
        elif len(tr) < n:
            tr = np.concatenate([tr, np.full(n - len(tr), None, dtype=object)])
        try:
            inv_cast = inv_arr.astype(tr.dtype, copy=False)
        except Exception:
            inv_cast = inv_arr
        return np.isin(tr, inv_cast)

    # fallback: try as indices
    try:
        idx = np.clip(inv_arr.astype(int), 0, max(0, n - 1))
        is_invalid[idx] = True
    except Exception:
        pass
    return is_invalid


def _make_valid_mask(on_abs, trial_ids, invalid_trials):
    return ~_normalize_invalid_selector(on_abs, trial_ids, invalid_trials)


# # =========================
# # Trial table reader (CSV/TSV)
# # =========================
# def _read_trial_table(path):
#     """
#     Robustly read a CSV/TSV trial table.

#     Modes:
#       - "by_index": first column integers (e.g., TrialNumber,...). Use duration from
#                     'duration' or fallback 'ResponseTime'.
#                     Label chars = first char of BlockName (if present).
#       - "by_onset": has explicit 'onset' column OR first column floats.
#                     Label chars = first char of 'category' (if present).
#     """
#     sep_used = None

#     def _try(sep):
#         nonlocal sep_used
#         try:
#             df = pd.read_csv(
#                 path, sep=sep, engine="python", encoding="utf-8-sig",
#                 skipinitialspace=True, dtype=str
#             )
#             sep_used = sep if sep in (",", "\t") else sep_used
#             return df
#         except Exception:
#             return None

#     # Try comma, then tab, then auto
#     df = _try(",")
#     if df is None:
#         df = _try("\t")
#     if df is None:
#         df = _try(None)
#     if df is None:
#         df = pd.read_csv(path, header=None, engine="python", encoding="utf-8-sig")

#     # If the file collapsed into one column, try splitting by commas
#     if df.shape[1] == 1:
#         col0 = df.iloc[:, 0].astype(str)
#         header_cell = df.columns[0]
#         split = col0.str.split(",", expand=True)
#         # drop trailing all-empty columns
#         empty_col = split.apply(lambda s: s.isna() | (s.astype(str).str.strip() == ""), axis=0).all()
#         split = split.loc[:, ~empty_col]

#         if isinstance(header_cell, str) and "," in header_cell:
#             header_cols = [c.strip() for c in header_cell.split(",")]
#             if split.shape[1] > len(header_cols):
#                 split = split.iloc[:, :len(header_cols)]
#             if split.shape[1] < len(header_cols):
#                 header_cols = header_cols[:split.shape[1]]
#             split.columns = header_cols
#             df = split
#         else:
#             maybe_header = [str(x).strip() for x in split.iloc[0].tolist()]
#             if split.shape[1] > len(maybe_header):
#                 split = split.iloc[:, :len(maybe_header)]
#             split.columns = maybe_header
#             df = split.iloc[1:].reset_index(drop=True)

#         sep_used = ","  # we split by commas

#     # cleanup
#     df = df.dropna(how="all")
#     df.columns = [str(c).strip() for c in df.columns]
#     dfl = df.rename(columns={c: c.lower() for c in df.columns})

#     # duration: prefer 'duration', fallback 'responsetime'/'response_time'
#     dur_col = None
#     if "duration" in dfl.columns:
#         dur_col = "duration"
#     elif "responsetime" in dfl.columns:
#         dur_col = "responsetime"
#     elif "response_time" in dfl.columns:
#         dur_col = "response_time"

#     duration_sec = None
#     if dur_col is not None:
#         duration_sec = pd.to_numeric(dfl[dur_col], errors="coerce").to_numpy()

#     # explicit onset column?
#     if "onset" in dfl.columns:
#         onset_sec = pd.to_numeric(dfl["onset"], errors="coerce").to_numpy()
#         label_chars = None
#         if "category" in dfl.columns:
#             s = dfl["category"].astype(str).str.strip()
#             label_chars = s.str[0].fillna("").str.lower().to_numpy()
#         return {
#             "mode": "by_onset",
#             "onset_sec": onset_sec,
#             "duration_sec": duration_sec,
#             "raw_df": df,
#             "source_path": path,
#             "duration_col_used": dur_col,
#             "sep_used": sep_used,
#             "label_chars": label_chars,
#         }

#     # infer from first column
#     label_chars = None
#     if df.shape[1] > 0:
#         first = df.columns[0]
#         s0 = pd.to_numeric(df[first], errors="coerce")
#         if s0.notna().all():
#             if np.all(np.equal(np.mod(s0, 1), 0)):  # integers -> by_index
#                 if "blockname" in dfl.columns:
#                     s = dfl["blockname"].astype(str).str.strip()
#                     label_chars = s.str[0].fillna("").str.lower().to_numpy()
#                 return {
#                     "mode": "by_index",
#                     "onset_sec": None,
#                     "duration_sec": duration_sec,
#                     "raw_df": df,
#                     "source_path": path,
#                     "duration_col_used": dur_col,
#                     "sep_used": sep_used,
#                     "label_chars": label_chars,
#                 }
#             else:  # floats -> by_onset
#                 onset_sec = s0.to_numpy(float)
#                 if "category" in dfl.columns:
#                     s = dfl["category"].astype(str).str.strip()
#                     label_chars = s.str[0].fillna("").str.lower().to_numpy()
#                 return {
#                     "mode": "by_onset",
#                     "onset_sec": onset_sec,
#                     "duration_sec": duration_sec,
#                     "raw_df": df,
#                     "source_path": path,
#                     "duration_col_used": dur_col,
#                     "sep_used": sep_used,
#                     "label_chars": label_chars,
#                 }

#     # fallback: try typical trial number column => by_index
#     for c in df.columns:
#         if c.lower() in ("trialnumber", "trial", "trial_num", "trial_index"):
#             trid = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int).to_numpy()
#             if "blockname" in dfl.columns:
#                 s = dfl["blockname"].astype(str).str.strip()
#                 label_chars = s.str[0].fillna("").str.lower().to_numpy()
#             return {
#                 "mode": "by_index",
#                 "onset_sec": None,
#                 "duration_sec": duration_sec,
#                 "raw_df": df,
#                 "source_path": path,
#                 "duration_col_used": dur_col,
#                 "sep_used": sep_used,
#                 "label_chars": label_chars,
#             }

#     # last resort
#     return {
#         "mode": "by_index",
#         "onset_sec": None,
#         "duration_sec": duration_sec,
#         "raw_df": df,
#         "source_path": path,
#         "duration_col_used": dur_col,
#         "sep_used": sep_used,
#         "label_chars": None,
#     }

def _read_manual_triggers(path, fs):
    """
    Load precomputed onsets/offsets from a TSV/CSV.

    Preferred columns (case-insensitive):
      - 'sample' (absolute onset sample)
      - 'sample_offsets' (absolute offset sample)
    Fallback (if samples missing):
      - 'onset' (seconds), 'onset_duration' (seconds) -> convert to samples
    Returns dict: {'on_abs': np.int64[], 'off_abs': np.int64[], 'raw_df': DataFrame}
    """
    import pandas as pd
    import numpy as np

    # robust read + single-column split if needed
    def _read_any(p):
        try:
            df = pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(p, engine="python", encoding="utf-8-sig")
        if df.shape[1] == 1:
            c0 = df.iloc[:, 0].astype(str)
            if ("\t" in c0.iloc[0]) or c0.str.contains("\t").any():
                split = c0.str.split("\t", expand=True)
            elif ("\\t" in c0.iloc[0]) or c0.str.contains(r"\\t").any():
                split = c0.str.split(r"\\t", expand=True)
            else:
                split = c0.str.split(",", expand=True)
            # header from first row
            header = [str(x).strip() for x in split.iloc[0].tolist()]
            split = split.iloc[1:].reset_index(drop=True)
            split.columns = header
            return split
        return df

    df = _read_any(path)
    cols = {c.lower(): c for c in df.columns}

    def _to_num(colname):
        return pd.to_numeric(df[cols[colname]], errors="coerce")

    on_abs = off_abs = None

    # Preferred: sample/sample_offsets present
    if "sample" in cols and "sample_offsets" in cols:
        on_abs  = _to_num("sample")
        off_abs = _to_num("sample_offsets")
        mask = on_abs.notna() & off_abs.notna()
        on_abs  = on_abs[mask].astype(np.int64).to_numpy()
        off_abs = off_abs[mask].astype(np.int64).to_numpy()
    else:
        # Fallback: onset/onset_duration in seconds
        onset_key = "onset" if "onset" in cols else None
        dur_key   = "onset_duration" if "onset_duration" in cols else ("duration" if "duration" in cols else None)
        if onset_key is None or dur_key is None:
            raise ValueError(f"Manual triggers TSV '{path}' lacks 'sample'+'sample_offsets' "
                             f"or 'onset'+'onset_duration'/'duration' columns.")
        onset_s = _to_num(onset_key)
        dur_s   = _to_num(dur_key)
        mask = onset_s.notna() & dur_s.notna()
        on_samp  = np.round(onset_s[mask].to_numpy(float) * float(fs)).astype(np.int64)
        off_samp = on_samp + np.round(dur_s[mask].to_numpy(float) * float(fs)).astype(np.int64)
        on_abs, off_abs = on_samp, off_samp

    return {"on_abs": on_abs, "off_abs": off_abs, "raw_df": df}



def _read_trial_table(path):
    import pandas as pd
    import numpy as np

    sep_used = None

    def _try(sep):
        nonlocal sep_used
        try:
            df = pd.read_csv(
                path, sep=sep, engine="python", encoding="utf-8-sig",
                skipinitialspace=True, dtype=str
            )
            sep_used = sep if sep in (",", "\t") else sep_used
            return df
        except Exception:
            return None

    # Try comma, then tab, then auto
    df = _try(",")
    if df is None:
        df = _try("\t")
    if df is None:
        df = _try(None)
    if df is None:
        # last resort: no header inference
        df = pd.read_csv(path, header=None, engine="python", encoding="utf-8-sig")

    # ---- single-merged-column fixups ----
    # ---- single-merged-column fixups ----
    if df.shape[1] == 1:
        col0 = df.iloc[:, 0].astype(str)
        header_cell = df.columns[0]  # may itself be a merged line

        # Detect real and escaped tabs in header or data
        header_has_tabs     = isinstance(header_cell, str) and ("\t" in header_cell)
        data_has_tabs       = col0.str.contains("\t").any()
        header_has_esc_tabs = isinstance(header_cell, str) and ("\\t" in header_cell)
        data_has_esc_tabs   = col0.str.contains(r"\\t").any()

        # Choose how to split the single column
        if header_has_tabs or data_has_tabs:
            split = col0.str.split("\t", expand=True)
            sep_used = "\t"
            header_sep = "\t"
        elif header_has_esc_tabs or data_has_esc_tabs:
            split = col0.str.split(r"\\t", expand=True)
            sep_used = "\t"
            header_sep = r"\\t"
        else:
            split = col0.str.split(",", expand=True)
            sep_used = ","
            header_sep = ","

        # Drop all-empty trailing columns
        empty_col = split.apply(lambda s: s.isna() | (s.astype(str).str.strip() == ""), axis=0).all()
        split = split.loc[:, ~empty_col]

        # Build header from header cell if it contains the chosen separator,
        # otherwise use the first row as header and drop it from the data.
        use_header_cell = (
            isinstance(header_cell, str) and
            ((header_sep == "\t"   and "\t"  in header_cell) or
             (header_sep == r"\\t" and "\\t" in header_cell) or
             (header_sep == ","    and ","   in header_cell))
        )

        if use_header_cell:
            header_cols = [c.strip() for c in header_cell.split(header_sep)]
            if split.shape[1] > len(header_cols):
                split = split.iloc[:, :len(header_cols)]
            if split.shape[1] < len(header_cols):
                header_cols = header_cols[:split.shape[1]]
            split.columns = header_cols
            df = split
        else:
            maybe_header = [str(x).strip() for x in split.iloc[0].tolist()]
            if split.shape[1] > len(maybe_header):
                split = split.iloc[:, :len(maybe_header)]
            split.columns = maybe_header
            df = split.iloc[1:].reset_index(drop=True)

        
        
#         header_has_esc_tabs = isinstance(header_cell, str) and "\\t" in header_cell
#         data_has_esc_tabs   = col0.str.contains(r"\\t").any()

#         if header_has_esc_tabs or data_has_esc_tabs:
#             split = col0.str.split(r"\\t", expand=True)

#             # drop all-empty trailing columns
#             empty_col = split.apply(lambda s: s.isna() | (s.astype(str).str.strip() == ""), axis=0).all()
#             split = split.loc[:, ~empty_col]

#             # If the header cell itself contains '\t', use it as header
#             if header_has_esc_tabs:
#                 header_cols = [c.strip() for c in header_cell.split("\\t")]
#                 if split.shape[1] > len(header_cols):
#                     split = split.iloc[:, :len(header_cols)]
#                 if split.shape[1] < len(header_cols):
#                     header_cols = header_cols[:split.shape[1]]
#                 split.columns = header_cols
#                 df = split
#             else:
#                 # Otherwise treat first row as header
#                 maybe_header = [str(x).strip() for x in split.iloc[0].tolist()]
#                 if split.shape[1] > len(maybe_header):
#                     split = split.iloc[:, :len(maybe_header)]
#                 split.columns = maybe_header
#                 df = split.iloc[1:].reset_index(drop=True)

#             # Treat these files as TSV-like for downstream logic
#             sep_used = "\t"

#         else:
#             # Case B: comma-separated text baked into one column → split by comma (existing behavior)
#             split = col0.str.split(",", expand=True)

#             empty_col = split.apply(lambda s: s.isna() | (s.astype(str).str.strip() == ""), axis=0).all()
#             split = split.loc[:, ~empty_col]

#             if isinstance(header_cell, str) and "," in header_cell:
#                 header_cols = [c.strip() for c in header_cell.split(",")]
#                 if split.shape[1] > len(header_cols):
#                     split = split.iloc[:, :len(header_cols)]
#                 if split.shape[1] < len(header_cols):
#                     header_cols = header_cols[:split.shape[1]]
#                 split.columns = header_cols
#                 df = split
#             else:
#                 maybe_header = [str(x).strip() for x in split.iloc[0].tolist()]
#                 if split.shape[1] > len(maybe_header):
#                     split = split.iloc[:, :len(maybe_header)]
#                 split.columns = maybe_header
#                 df = split.iloc[1:].reset_index(drop=True)

#             sep_used = ","

    # ---- normalize column names & infer fields ----
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    dfl = df.rename(columns={c: c.lower() for c in df.columns})

    # duration: prefer 'duration', fallback 'responsetime'/'response_time'
    dur_col = None
    for c in dfl.columns:
        if c == "duration":
            dur_col = c; break
    if dur_col is None:
        for c in dfl.columns:
            if c in ("responsetime", "response_time"):
                dur_col = c; break

    duration_sec = None
    if dur_col is not None:
        duration_sec = pd.to_numeric(dfl[dur_col], errors="coerce").to_numpy()

    # explicit onset?
    if "onset" in dfl.columns:
        onset_sec = pd.to_numeric(dfl["onset"], errors="coerce").to_numpy()
        label_chars = None
        # for TSV-like, prefer 'category' first char if present
        if "category" in dfl.columns:
            s = dfl["category"].astype(str).str.strip()
            label_chars = s.str[0].fillna("").str.lower().to_numpy()

        return {
            "mode": "by_onset",
            "onset_sec": onset_sec,
            "duration_sec": duration_sec,
            "trial_ids": None,
            "raw_df": df,
            "source_path": path,
            "duration_col_used": dur_col,
            "sep_used": sep_used,
            "label_chars": label_chars,
        }

    # decide by first column (index-style files) if no explicit onset
    label_chars = None
    if df.shape[1] > 0:
        first = df.columns[0]
        s0 = pd.to_numeric(df[first], errors="coerce")
        if s0.notna().all():
            if np.all(np.equal(np.mod(s0, 1), 0)):  # integers => index-style
                if sep_used == "," and "blockname" in dfl.columns:
                    s = dfl["blockname"].astype(str).str.strip()
                    label_chars = s.str[0].fillna("").str.lower().to_numpy()
                return {
                    "mode": "by_index",
                    "onset_sec": None,
                    "duration_sec": duration_sec,
                    "trial_ids": s0.astype(int).to_numpy(),
                    "raw_df": df,
                    "source_path": path,
                    "duration_col_used": dur_col,
                    "sep_used": sep_used,
                    "label_chars": label_chars,
                }
            else:
                # floats => onset-style
                onset_sec = s0.to_numpy(float)
                if "category" in dfl.columns:
                    s = dfl["category"].astype(str).str.strip()
                    label_chars = s.str[0].fillna("").str.lower().to_numpy()
                return {
                    "mode": "by_onset",
                    "onset_sec": onset_sec,
                    "duration_sec": duration_sec,
                    "trial_ids": None,
                    "raw_df": df,
                    "source_path": path,
                    "duration_col_used": dur_col,
                    "sep_used": sep_used,
                    "label_chars": label_chars,
                }

    # fallback: typical trial number column
    for c in df.columns:
        if c.lower() in ("trialnumber", "trial", "trial_num", "trial_index"):
            trid = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int).to_numpy()
            if sep_used == "," and "blockname" in dfl.columns:
                s = dfl["blockname"].astype(str).str.strip()
                label_chars = s.str[0].fillna("").str.lower().to_numpy()
            return {
                "mode": "by_index",
                "onset_sec": None,
                "duration_sec": duration_sec,
                "trial_ids": trid,
                "raw_df": df,
                "source_path": path,
                "duration_col_used": dur_col,
                "sep_used": sep_used,
                "label_chars": label_chars,
            }

    # last resort
    return {
        "mode": "by_index",
        "onset_sec": None,
        "duration_sec": duration_sec,
        "trial_ids": None,
        "raw_df": df,
        "source_path": path,
        "duration_col_used": dur_col,
        "sep_used": sep_used,
        "label_chars": None,
    }



# =========================
# Plotting: PD detections only (green/red)
# =========================
def _plot_pd_detections(*, pd_full, fs, i0, i1, on_abs, off_abs,
                        trial_ids=None, invalid_trials=None, title=None, plot_kwargs=None):
    plot_kwargs = {} if plot_kwargs is None else dict(plot_kwargs)
    t = np.arange(i0, i1) / fs
    y = pd_full[i0:i1]

    fig, ax = plt.subplots(figsize=(30, 4.5))
    ax.plot(t, y, linewidth=1.0, **plot_kwargs)
    ax.locator_params(axis='x', nbins=50)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Photodiode (a.u.)")
    if title: ax.set_title(title)

    on_abs = np.asarray(on_abs); off_abs = np.asarray(off_abs)
    n_tr = len(on_abs)
    if n_tr == 0:
        plt.show(); return

    y_min, y_max = np.nanmin(y), np.nanmax(y)
    y_span = y_max - y_min if y_max > y_min else 1.0
    y_top = y_max + 0.05 * y_span

    is_invalid = _normalize_invalid_selector(on_abs, trial_ids, invalid_trials)

    for k in range(n_tr):
        onset_samp = on_abs[k]; onset_t = onset_samp / fs
        if not (i0 <= onset_samp < i1): continue

        vcolor = "red" if is_invalid[k] else "tab:green"
        shade_color = (1.0, 0.8, 0.8, 0.25) if is_invalid[k] else (0.8, 1.0, 0.8, 0.22)

        ax.axvline(onset_t, linestyle="--", linewidth=1.25, color=vcolor, alpha=0.9)
        ax.plot([onset_t], [y_top], marker='v', ms=7, color=vcolor)
        ax.locator_params(axis='x', nbins=1000)


        label = f"{k}:{trial_ids[k]}" if (trial_ids is not None and k < len(trial_ids)) else f"{k}"
        ax.text(onset_t, y_top, label, ha="center", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=vcolor, lw=1, alpha=0.9))

        if k < len(off_abs):
            off_samp = off_abs[k]; off_t = off_samp / fs
            left = max(onset_t, i0 / fs); right = min(off_t, i1 / fs) if off_samp > onset_samp else left
            if right > left:
                ax.axvspan(left, right, color=shade_color, linewidth=0)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color="tab:green", lw=2, linestyle="--", label="Valid PD onset"),
        Line2D([0],[0], color="red",       lw=2, linestyle="--", label="Invalid PD onset"),
    ], loc="upper right", framealpha=0.9)

    ax.set_ylim(y_min - 0.05*y_span, y_top + 0.1*y_span)
    ax.grid(True, alpha=0.25); plt.tight_layout(); plt.show()


# =========================
# Plotting: PD + extra table overlays (orange)
# =========================
def _plot_pd_detections_with_extra(
    *, pd_full, fs, i0, i1, on_abs, off_abs,
    trial_ids=None, invalid_trials=None, extra=None, title=None, plot_kwargs=None,
    verbose=True, preview_rows=5
):
    """
    Draws PD detections (green/red).
    If 'extra' is provided:
      - CSV/by_index: use duration to draw orange shading **anchored at PD OFFSET**.
      - TSV/by_onset: draw shifted orange onset lines (reference) and also draw orange
        duration shading **anchored at PD OFFSET** using the durations (index-wise).
    Returns a metrics dict with dur_samples, anchor_abs (offsets), duration_end_abs, tags, etc.
    """
    plot_kwargs = {} if plot_kwargs is None else dict(plot_kwargs)
    t = np.arange(i0, i1) / fs
    y = pd_full[i0:i1]

    fig, ax = plt.subplots(figsize=(30*8, 3))
    ax.plot(t, y, linewidth=1.0, **plot_kwargs)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Photodiode (a.u.)")
    if title: ax.set_title(title)

    on_abs = np.asarray(on_abs); off_abs = np.asarray(off_abs)
    n_tr = len(on_abs)
    if n_tr == 0:
        if verbose: print("[plot] No PD detections -> nothing to plot.")
        plt.show(); 
        return {"mode": None, "dur_samples": None, "anchor_abs": None, "duration_end_abs": None, "tags": None}

    y_min, y_max = np.nanmin(y), np.nanmax(y)
    y_span = y_max - y_min if y_max > y_min else 1.0
    y_top = y_max + 0.05 * y_span

    is_invalid = _normalize_invalid_selector(on_abs, trial_ids, invalid_trials)

    # ----- PD detections (green/red) -----
    for k in range(n_tr):
        onset_samp = on_abs[k]; onset_t = onset_samp / fs
        if not (i0 <= onset_samp < i1): continue

        vcolor = "red" if is_invalid[k] else "tab:green"
        shade_color = (1.0, 0.8, 0.8, 0.25) if is_invalid[k] else (0.8, 1.0, 0.8, 0.22)

        ax.axvline(onset_t, linestyle="--", linewidth=1, color=vcolor, alpha=0.8)
        ax.plot([onset_t], [y_top], marker='v', ms=7, color=vcolor)
        ax.locator_params(axis='x', nbins=1000)

        label = f"{k}:{trial_ids[k]}" if (trial_ids is not None and k < len(trial_ids)) else f"{k}"
        ax.text(onset_t, y_top, label, ha="center", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=vcolor, lw=0.8, alpha=0.9))

        if k < len(off_abs):
            off_samp = off_abs[k]; off_t = off_samp / fs
            left = max(onset_t, i0 / fs); right = min(off_t, i1 / fs) if off_samp > onset_samp else left
            if right > left:
                ax.axvspan(left, right, color=shade_color, linewidth=0)

    # ----- Orange overlays from extra -----
    mode = None
    sep_used = None
    label_chars = None
    duration_sec = None
    onset_sec = None
    orange_on = None
    orange_end = None
    dur_samp = None
    anchor = None
    shade_orange = (1.0, 0.6, 0.0, 0.22)  # RGBA

    if extra is not None:
        mode = extra.get("mode")
        sep_used = extra.get("sep_used")
        label_chars = extra.get("label_chars")
        duration_sec = extra.get("duration_sec")
        onset_sec = extra.get("onset_sec")
        src = extra.get("source_path", "<unknown>")
        dur_col = extra.get("duration_col_used")

        if verbose:
            print(f"[extra] source: {src}")
            print(f"[extra] mode: {mode} | fs={fs:.6g} | window_s=[{i0/fs:.3f},{i1/fs:.3f}]")
            if hasattr(extra.get("raw_df", None), "head"):
                print("[extra] head:"); print(extra["raw_df"].head(preview_rows))
            print(f"[extra] columns: {list(extra['raw_df'].columns) if 'raw_df' in extra else 'n/a'}")
            print(f"[extra] duration_col_used: {dur_col}")
            if duration_sec is not None:
                print(f"[extra] duration non-NaN count: {int(np.isfinite(duration_sec).sum())}")

        # Helper: char tag for index k
        def tag_char(k):
            if label_chars is not None and len(label_chars) > k and isinstance(label_chars[k], str) and len(label_chars[k]) > 0:
                return label_chars[k][0].lower()
            # fallback: CSV->'p', TSV->'a' (arbitrary), else 'd'
            return 'p' if sep_used == ',' else ('a' if sep_used == '\t' else 'd')

        if mode == "by_index":
            if duration_sec is not None:
                n_map = min(len(duration_sec), len(off_abs))
                d_ok = np.asarray(duration_sec[:n_map], float)
                dur_samp = np.round(d_ok * fs).astype(int)
                anchor = off_abs[:n_map].astype(int)                    # **PD offset** anchor
                orange_end = anchor + dur_samp

                drew_end = False
                for k in range(n_map):
                    s_off = anchor[k]; s_end = orange_end[k]
                    if s_end <= s_off: continue
                    # end line
                    if i0 <= s_end < i1:
                        ax.axvline(s_end / fs, linestyle="-", linewidth=1.25, color="orange", alpha=0.95)
                        drew_end = True
                    # shading (offset -> end)
                    left = max(s_off, i0) / fs
                    right = min(s_end, i1) / fs
                    if right > left:
                        ax.axvspan(left, right, color=shade_orange, linewidth=0)
                        # tag atop
                        mid = 0.5 * (left + right)
                        ax.text(mid, y_top, f"{tag_char(k)}{k+1}",
                                ha="center", va="bottom", fontsize=7,
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="orange", lw=1, alpha=0.9))
                if verbose and orange_end is not None and len(orange_end):
                    print(f"[extra] orange_end samples (first 5): {orange_end[:5]}")
                    print(f"[extra] orange_end seconds (first 5): {np.round((orange_end[:5]/fs), 6)}")
                if verbose and not drew_end:
                    print("[extra] orange_end computed but outside window -> nothing drawn.")
            else:
                if verbose: print("[extra] No duration/ResponseTime column -> no orange shading.")

        elif mode == "by_onset":
            # draw shifted file onsets (reference only)
            if onset_sec is not None and len(onset_sec) > 0:
                onset_sec = np.asarray(onset_sec, float)
                shift_samp = int(on_abs[0] - round(onset_sec[0] * fs))
                orange_on = np.round(onset_sec * fs).astype(int) + shift_samp
                drew_on = False
                for s in orange_on:
                    if i0 <= s < i1:
                        ax.axvline(s / fs, linestyle="-.", linewidth=1.25, color="orange", alpha=0.95)
                        drew_on = True
                if verbose and not drew_on:
                    print("[extra] orange_on computed but outside window -> nothing drawn.")
            else:
                if verbose: print("[extra] No usable onset column -> no orange file onsets.")

            # duration shading still anchored at PD offset (index-wise)
            if duration_sec is not None:
                n_map = min(len(duration_sec), len(off_abs))
                d_ok = np.asarray(duration_sec[:n_map], float)
                dur_samp = np.round(d_ok * fs).astype(int)
                anchor = off_abs[:n_map].astype(int)                    # **PD offset** anchor
                orange_end = anchor + dur_samp

                drew_end = False
                for k in range(n_map):
                    s_off = anchor[k]; s_end = orange_end[k]
                    if s_end <= s_off: continue
                    if i0 <= s_end < i1:
                        ax.axvline(s_end / fs, linestyle="-", linewidth=1.25, color="orange", alpha=0.95)
                        drew_end = True
                    left = max(s_off, i0) / fs
                    right = min(s_end, i1) / fs
                    if right > left:
                        ax.axvspan(left, right, color=shade_orange, linewidth=0)
                        mid = 0.5 * (left + right)
                        ax.text(mid, y_top, f"{tag_char(k)}{k+1}",
                                ha="center", va="bottom", fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="orange", lw=1, alpha=0.9))
                if verbose and orange_end is not None and len(orange_end):
                    print(f"[extra] orange_end samples (first 5): {orange_end[:5]}")
                    print(f"[extra] orange_end seconds (first 5): {np.round((orange_end[:5]/fs), 6)}")
                if verbose and not drew_end:
                    print("[extra] orange_end computed but outside window -> nothing drawn.")
            else:
                if verbose: print("[extra] No duration column -> no orange shading.")

        # Legend proxies incl. orange
        from matplotlib.lines import Line2D
        elems = [
            Line2D([0],[0], color="tab:green", lw=2, linestyle="--", label="Valid PD onset"),
            Line2D([0],[0], color="red",       lw=2, linestyle="--", label="Invalid PD onset"),
        ]
        if orange_on is not None:
            elems.append(Line2D([0],[0], color="orange", lw=2, linestyle="-.", label="File onset (shifted)"))
        if orange_end is not None:
            elems.append(Line2D([0],[0], color="orange", lw=2, linestyle="-",  label="File end (offset+duration)"))
        elems.append(Line2D([0],[0], color="orange", lw=8, alpha=0.3, label="Duration shading"))
        ax.legend(handles=elems, loc="upper right", framealpha=0.9)
        ax.locator_params(axis='x', nbins=1000)

    else:
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0],[0], color="tab:green", lw=2, linestyle="--", label="Valid PD onset"),
            Line2D([0],[0], color="red",       lw=2, linestyle="--", label="Invalid PD onset"),
        ], loc="upper right", framealpha=0.9)

    ax.set_ylim(y_min - 0.05*y_span, y_top + 0.1*y_span)
    ax.locator_params(axis='x', nbins=1000)

    ax.grid(True, alpha=0.25); plt.tight_layout(); plt.show()

    return {
        'mode': mode,
        'sep_used': sep_used,
        'dur_samples': dur_samp,
        'anchor': 'pd_offset' if anchor is not None else None,
        'anchor_abs': anchor,
        'duration_end_abs': orange_end,
        'tags': [ (label_chars[k][0].lower() if label_chars is not None and len(label_chars)>k and isinstance(label_chars[k], str) and label_chars[k] else 'd') + str(k+1)
                  for k in range(len(anchor)) ] if anchor is not None else None
    }


# =========================
# Public API: monopolar PD extractor
# =========================
# def get_trigger_indexes_photodiode(
#     *,
#     raw_signals,
#     sampling_rate,
#     channel_names,
#     trig_name,              # str: name of the PD channel (monopolar)
#     time_range,             # (t0, t1) seconds; use -1 for end
#     threshold_val=0.8,
#     flip_trigs=False,
#     skip_samples=50,
#     do_plot=True,

#     # Plotting / selection
#     trial_ids=None,
#     invalid_trials=None,
#     ignore_invalid=False,
#     plot_title=None,
#     plot_kwargs=None,

#     # Extras
#     fake_trials=None,               # indices or IDs to drop entirely
#     extra_table_path=None,          # CSV/TSV path for orange overlays
#     extra_verbose=True,
#     extra_preview_rows=5,
#     return_extra_metrics=False,     # if True -> append metrics dict to returns
# ):
#     """
#     Monopolar-only PD trigger extraction.
#     Returns
#     -------
#     on_abs, off_abs
#     (+ valid_on, valid_off if ignore_invalid=True & invalid_trials provided)
#     (+ metrics dict if return_extra_metrics=True)
#     """
#     fs = float(sampling_rate)
#     t0, t1 = time_range
#     i0 = max(0, int(round(t0 * fs)))
#     i1 = raw_signals.shape[0] if (t1 == -1 or t1 is None) else min(int(round(t1 * fs)), raw_signals.shape[0])

#     # resolve PD channel (MONOPOLAR ONLY)
#     name_to_idx = {str(n): i for i, n in enumerate(channel_names)}
#     if isinstance(trig_name, (tuple, list)):
#         raise ValueError("This module is monopolar-only. 'trig_name' must be a single channel name (str).")
#     i_ch = name_to_idx[str(trig_name)]
#     pd_full = raw_signals[:, i_ch]
#     trig_label = str(trig_name)

#     # detect
#     on_abs, off_abs = _pd_detect_deriv_core(
#         pd_full,
#         t_start=i0, t_end=i1, sampling_rate=fs,
#         threshold_val=threshold_val, flip_trigs=flip_trigs,
#         skip_samples=skip_samples, do_plot=False
#     )
#     on_abs = np.asarray(on_abs, dtype=int)
#     off_abs = np.asarray(off_abs, dtype=int)

#     # remove fake trials (entirely)
#     if fake_trials is not None and len(on_abs):
#         n_on = len(on_abs)
#         # align trial_ids to n_on before building mask
#         trial_ids_arr = None
#         if trial_ids is not None:
#             trial_ids_arr = np.asarray(trial_ids, dtype=object)
#             if len(trial_ids_arr) > n_on:
#                 trial_ids_arr = trial_ids_arr[:n_on]
#             elif len(trial_ids_arr) < n_on:
#                 pad = np.full(n_on - len(trial_ids_arr), None, dtype=object)
#                 trial_ids_arr = np.concatenate([trial_ids_arr, pad])

#         fake_mask = _normalize_invalid_selector(on_abs, trial_ids_arr, fake_trials)  # True=fake
#         if len(fake_mask) != n_on:
#             fake_mask = np.asarray(fake_mask, dtype=bool)[:n_on]
#             if fake_mask.size < n_on:
#                 fake_mask = np.pad(fake_mask, (0, n_on - fake_mask.size), constant_values=False)

#         keep = ~fake_mask
#         on_abs = on_abs[keep]
#         if len(off_abs) == n_on:
#             off_abs = off_abs[keep]
#         if trial_ids_arr is not None:
#             trial_ids = trial_ids_arr[keep].tolist()

#     # parse extra table if provided
#     extra = None
#     if extra_table_path:
#         extra = _read_trial_table(extra_table_path)

#     # plot & metrics
#     metrics = None
#     if do_plot:
#         title = plot_title if plot_title is not None else f"Photodiode ({trig_label}) | {t0:.3f}s–{(i1/fs):.3f}s"
#         metrics = _plot_pd_detections_with_extra(
#             pd_full=pd_full, fs=fs, i0=i0, i1=i1,
#             on_abs=on_abs, off_abs=off_abs,
#             trial_ids=trial_ids, invalid_trials=invalid_trials,
#             extra=extra, title=title, plot_kwargs=plot_kwargs,
#             verbose=extra_verbose, preview_rows=extra_preview_rows
#         )

#     # optional valid-only returns
#     if ignore_invalid and invalid_trials is not None and len(on_abs):
#         valid_mask = ~_normalize_invalid_selector(on_abs, trial_ids, invalid_trials)
#         valid_on = on_abs[valid_mask]
#         if len(off_abs) == len(on_abs):
#             valid_off = off_abs[valid_mask]
#         else:
#             valid_off = off_abs[:valid_mask.sum()]
#         return (on_abs, off_abs, valid_on, valid_off, metrics) if return_extra_metrics else (on_abs, off_abs, valid_on, valid_off)

#     return (on_abs, off_abs, metrics) if return_extra_metrics else (on_abs, off_abs)


def get_trigger_indexes_photodiode(
    *,
    raw_signals,
    sampling_rate,
    channel_names,
    trig_name,              # str: name of the PD channel (monopolar)
    time_range,             # (t0, t1) seconds; use -1 for end
    threshold_val=0.8,
    flip_trigs=False,
    skip_samples=50,
    do_plot=True,

    # Plotting / selection
    trial_ids=None,
    invalid_trials=None,
    ignore_invalid=False,
    plot_title=None,
    plot_kwargs=None,

    # Extras
    fake_trials=None,               # indices or IDs to drop entirely
    extra_table_path=None,          # CSV/TSV path for orange overlays (task logs)
    extra_verbose=True,
    extra_preview_rows=5,
    return_extra_metrics=False,     # if True -> append metrics dict to returns

    # NEW: manual PD triggers (bypass detection)
    manual_trigs_path=None,         # TSV with columns: sample, sample_offsets (preferred)
                                    # or onset, onset_duration (seconds)
):
    """
    Monopolar-only PD trigger extraction.

    Modes
    -----
    1) Auto-detect (default): run the derivative-based detector on the PD channel.
    2) Manual mode (if manual_trigs_path is provided): SKIP detection and use the
       provided 'sample' (onsets) and 'sample_offsets' (offsets) as PD triggers.

    Returns
    -------
    on_abs, off_abs
    (+ valid_on, valid_off if ignore_invalid=True & invalid_trials provided)
    (+ metrics dict if return_extra_metrics=True)

    Notes
    -----
    - Orange duration overlay (from extra_table_path) is computed w.r.t. *PD offsets*.
      In manual mode, that means w.r.t. the offsets from your manual TSV.
    """

    import numpy as np
    import pandas as pd

    fs = float(sampling_rate)
    t0, t1 = time_range
    i0 = max(0, int(round(t0 * fs)))
    i1 = raw_signals.shape[0] if (t1 == -1 or t1 is None) else min(int(round(t1 * fs)), raw_signals.shape[0])

    # ---------- resolve PD channel (MONOPOLAR ONLY) ----------
    name_to_idx = {str(n): i for i, n in enumerate(channel_names)}
    if isinstance(trig_name, (tuple, list)):
        raise ValueError("This module is monopolar-only. 'trig_name' must be a single channel name (str).")
    i_ch = name_to_idx[str(trig_name)]
    pd_full = raw_signals[:, i_ch]
    trig_label = str(trig_name)

    # ---------- helper: read manual triggers ----------
    def _read_manual_pd_trigs(path, fs_hz):
        """
        Accepts TSV/CSV. Prefers integer 'sample' and 'sample_offsets'.
        Falls back to seconds 'onset' and 'onset_duration'.
        Returns: on_abs (int array), off_abs (int array)
        """
        # robust read (utf-8-sig; auto sep fallback)
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
        except Exception:
            # last resort
            df = pd.read_csv(path, sep="\t", engine="python", encoding="utf-8-sig", dtype=str)

        # normalize colnames
        df.columns = [str(c).strip().lower() for c in df.columns]
        # strip whitespace
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip()

        on_abs = off_abs = None

        # Preferred path: sample/sample_offsets exist (and numeric)
        has_samples = ("sample" in df.columns)
        has_offsamp = ("sample_offsets" in df.columns)
        if has_samples and has_offsamp:
            on_abs  = pd.to_numeric(df["sample"], errors="coerce").dropna().astype(int).to_numpy()
            off_abs = pd.to_numeric(df["sample_offsets"], errors="coerce").dropna().astype(int).to_numpy()
            # length reconcile
            n = min(len(on_abs), len(off_abs))
            on_abs, off_abs = on_abs[:n], off_abs[:n]
            return on_abs, off_abs

        # Fallback: onset/onset_duration in seconds
        has_onset_s  = ("onset" in df.columns)
        has_dur_s    = ("onset_duration" in df.columns) or ("duration" in df.columns)
        if has_onset_s and has_dur_s:
            dur_col = "onset_duration" if ("onset_duration" in df.columns) else "duration"
            onset_s = pd.to_numeric(df["onset"], errors="coerce").to_numpy()
            dur_s   = pd.to_numeric(df[dur_col], errors="coerce").to_numpy()
            valid   = np.isfinite(onset_s) & np.isfinite(dur_s)
            onset_s = onset_s[valid]; dur_s = dur_s[valid]
            on_abs  = np.round(onset_s * fs_hz).astype(int)
            # We *derive* offsets as: sample_offsets = sample + (onset_duration*fs)
            off_abs = on_abs + np.round(dur_s * fs_hz).astype(int)
            n = min(len(on_abs), len(off_abs))
            on_abs, off_abs = on_abs[:n], off_abs[:n]
            return on_abs, off_abs

        # If neither present, raise
        raise ValueError(
            "Manual triggers file must contain either "
            "'sample' & 'sample_offsets' (preferred) or 'onset' & 'onset_duration' (seconds)."
        )

    # ---------- choose mode: manual vs detection ----------
    manual_mode = manual_trigs_path is not None and str(manual_trigs_path).strip() != ""
    if manual_mode:
        # BYPASS detection completely
        on_abs, off_abs = _read_manual_pd_trigs(manual_trigs_path, fs)
        # Optional: restrict to time window [i0, i1) to match plotting/view bounds
        # (We do not drop out-of-window here; plotting clamps visually.)
        if extra_verbose:
            print(f"[manual] Using manual PD triggers from: {manual_trigs_path}")
            print(f"[manual] n_on={len(on_abs)}, n_off={len(off_abs)}")
    else:
        # ---------- auto-detect (legacy path) ----------
        on_abs, off_abs = _pd_detect_deriv_core(
            pd_full,
            t_start=i0, t_end=i1, sampling_rate=fs,
            threshold_val=threshold_val, flip_trigs=flip_trigs,
            skip_samples=skip_samples, do_plot=False
        )
        on_abs = np.asarray(on_abs, dtype=int)
        off_abs = np.asarray(off_abs, dtype=int)

    # ---------- remove fake trials (entirely) ----------
    if fake_trials is not None and len(on_abs):
        n_on = len(on_abs)
        # align trial_ids to n_on before building mask (truncate/pad)
        trial_ids_arr = None
        if trial_ids is not None:
            trial_ids_arr = np.asarray(trial_ids, dtype=object)
            if len(trial_ids_arr) > n_on:
                trial_ids_arr = trial_ids_arr[:n_on]
            elif len(trial_ids_arr) < n_on:
                pad = np.full(n_on - len(trial_ids_arr), None, dtype=object)
                trial_ids_arr = np.concatenate([trial_ids_arr, pad])

        fake_mask = _normalize_invalid_selector(on_abs, trial_ids_arr, fake_trials)  # True=fake
        if len(fake_mask) != n_on:
            fake_mask = np.asarray(fake_mask, dtype=bool)[:n_on]
            if fake_mask.size < n_on:
                fake_mask = np.pad(fake_mask, (0, n_on - fake_mask.size), constant_values=False)

        keep = ~fake_mask
        on_abs = on_abs[keep]
        if len(off_abs) == n_on:
            off_abs = off_abs[keep]
        if trial_ids_arr is not None:
            trial_ids = trial_ids_arr[keep].tolist()

    # ---------- parse extra task log (for orange overlays/metrics) ----------
    extra = None
    if extra_table_path:
        extra = _read_trial_table(extra_table_path)

    # ---------- plot & metrics ----------
    metrics = None
    if do_plot:
        mode_note = "Manual PD" if manual_mode else "Photodiode"
        title = plot_title if plot_title is not None else f"{mode_note} ({trig_label}) | {t0:.3f}s–{(i1/fs):.3f}s"
        metrics = _plot_pd_detections_with_extra(
            pd_full=pd_full, fs=fs, i0=i0, i1=i1,
            on_abs=on_abs, off_abs=off_abs,
            trial_ids=trial_ids, invalid_trials=invalid_trials,
            extra=extra, title=title, plot_kwargs=plot_kwargs,
            verbose=extra_verbose, preview_rows=extra_preview_rows
        )

    # ---------- optional valid-only returns ----------
    if ignore_invalid and invalid_trials is not None and len(on_abs):
        valid_mask = ~_normalize_invalid_selector(on_abs, trial_ids, invalid_trials)
        valid_on = on_abs[valid_mask]
        if len(off_abs) == len(on_abs):
            valid_off = off_abs[valid_mask]
        else:
            valid_off = off_abs[:valid_mask.sum()]
        return (on_abs, off_abs, valid_on, valid_off, metrics) if return_extra_metrics else (on_abs, off_abs, valid_on, valid_off)

    return (on_abs, off_abs, metrics) if return_extra_metrics else (on_abs, off_abs)


# =========================
# Saver (per condition; includes trial_end etc.)
# =========================
def save_onsets_offsets_by_condition(
    *,
    patient_id,
    block_name,
    onsets, offsets,
    sampling_rate,
    trial_ids,            # labels used to split into picture/audio/reading (aliases allowed)
    out_dir,
    invalid_indices=None,
    include_invalid=False,
    cond_alias=None,      # e.g. {"picture_naming":"pict","auditory_naming_FRE":"audi","reading_completion":"read"}
    trigger_label="triggerPD",
    date_str=None,

    # optional per-trial metadata (aligned with detection order)
    trial_end_abs=None,   # absolute sample of end = PD offset + duration(samples)
    condition_name=None,  # BlockName (CSV) or category (TSV)
    resp_accuracy=None,   # ResponseAccuracy (CSV) or response_type (TSV)
    trial_idx=None,       # StimNumber (CSV) or exemplar (TSV)
):
    """
    Writes one TSV per condition with columns:
      onset  onset_duration  sample  sample_offsets  trial_end  condition_name  resp_accuracy  trial_idx
    """
    os.makedirs(out_dir, exist_ok=True)
    fs = float(sampling_rate)
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")

    # normalize condition labels -> short keys for file split
    def _norm_short(lbl: str) -> str:
        s = str(lbl).strip().lower()
        if cond_alias and s in cond_alias:
            s = cond_alias[s]
        if s in ("picture", "picture_naming", "pict"):   return "pict"
        if s in ("auditory", "auditory_naming", "auditory_naming_fre", "audi"): return "audi"
        if s in ("reading", "reading_completion", "read"): return "read"
        return s

    short_to_full = {"pict": "picture", "audi": "audio", "read": "reading"}

    on = np.asarray(onsets, dtype=int)
    off = np.asarray(offsets, dtype=int)

    # pair onset->next offset
    i = j = 0
    paired = []
    while i < len(on) and j < len(off):
        if off[j] <= on[i]:
            j += 1; continue
        paired.append((i, j)); i += 1; j += 1
    if not paired:
        raise ValueError("No valid onset→offset pairs found (all offsets <= onsets).")

    paired_on  = np.array([on[i0]  for (i0, j0) in paired], dtype=int)
    paired_off = np.array([off[j0] for (i0, j0) in paired], dtype=int)
    n_pairs    = len(paired_on)

    trial_ids = list(trial_ids or [])
    if len(trial_ids) < n_pairs:
        raise ValueError(f"trial_ids length ({len(trial_ids)}) < number of paired trials ({n_pairs}).")
    labels_short = np.array([_norm_short(trial_ids[k]) for k in range(n_pairs)], dtype=object)

    # align metadata to n_pairs
    def _safe_slice(arr, n, cast=None):
        if arr is None: return None
        a = np.asarray(arr)
        if a.ndim != 1: a = a.ravel()
        a = a[:n]
        if cast is not None:
            try: a = a.astype(cast)
            except Exception: pass
        return a

    trial_end_abs = _safe_slice(trial_end_abs, n_pairs, int)
    condition_name = _safe_slice(condition_name, n_pairs, object)
    resp_accuracy  = _safe_slice(resp_accuracy,  n_pairs, object)
    trial_idx      = _safe_slice(trial_idx,      n_pairs, object)

    # drop invalid paired indices (after pairing)
    keep_mask = np.ones(n_pairs, dtype=bool)
    if invalid_indices and not include_invalid:
        inv = np.clip(np.asarray(invalid_indices, dtype=int), 0, n_pairs - 1)
        keep_mask[inv] = False

    paired_on      = paired_on[keep_mask]
    paired_off     = paired_off[keep_mask]
    labels_short   = labels_short[keep_mask]
    if trial_end_abs is not None: trial_end_abs = trial_end_abs[keep_mask]
    if condition_name is not None: condition_name = condition_name[keep_mask]
    if resp_accuracy  is not None: resp_accuracy  = resp_accuracy[keep_mask]
    if trial_idx      is not None: trial_idx      = trial_idx[keep_mask]

    # writer
    def _write_cond(short_key):
        full_name = short_to_full.get(short_key, short_key)
        mask = (labels_short == short_key)
        if not np.any(mask): return None
        fname = os.path.join(
            out_dir,
            f"{patient_id}_{block_name}_{full_name}__{trigger_label}_{date_str}.tsv"
        )
        with open(fname, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow([
                "onset","onset_duration","sample","sample_offsets",
                "trial_end","condition_name","resp_accuracy","trial_idx"
            ])
            on_m  = paired_on[mask]; off_m = paired_off[mask]
            te_m  = (trial_end_abs[mask] if trial_end_abs is not None else np.full(on_m.shape, "", dtype=object))
            cn_m  = (condition_name[mask] if condition_name is not None else np.full(on_m.shape, "", dtype=object))
            ra_m  = (resp_accuracy[mask]  if resp_accuracy  is not None else np.full(on_m.shape, "", dtype=object))
            ti_m  = (trial_idx[mask]      if trial_idx      is not None else np.full(on_m.shape, "", dtype=object))
            for k in range(len(on_m)):
                on_samp  = int(on_m[k]); off_samp = int(off_m[k])
                onset_s  = on_samp / fs
                dur_s    = (off_samp - on_samp) / fs
                te = te_m[k]
                te_out = int(te) if (isinstance(te, (int, np.integer)) or (isinstance(te, str) and te.isdigit())) else (int(te) if hasattr(te, '__int__') else te)
                w.writerow([onset_s, dur_s, on_samp, off_samp, te_out, cn_m[k], ra_m[k], ti_m[k]])
        return fname

    files = {
        "picture": _write_cond("pict"),
        "audio":   _write_cond("audi"),
        "reading": _write_cond("read"),
    }
    return {"n_pairs": int(n_pairs), "files": files}


def parse_and_save(pid, patient_id, on_abs, off_abs, metrics,
                   sampling_rate, save_path, block_name, exp_file, trial_ids, trig_label,
                   *, cond_alias=None):
    """Parse PD extraction results and save per-condition TSV files."""
    cond_alias = cond_alias or {}
    if exp_file and os.path.exists(exp_file):
        extra = _read_trial_table(exp_file)
        dfl   = extra["raw_df"].rename(columns=str.lower)
    else:
        dfl = None

    def _pick(cols):
        for c in cols:
            if dfl is not None and c in dfl:
                return dfl[c].astype(str).to_numpy()
        return None

    condition_name = _pick(["category", "blockname"])
    resp_accuracy  = _pick(["response_type", "responseaccuracy"])
    trial_idx_col  = _pick(["exemplar", "stimnumber"])
    trial_ids_for_save = (
        trial_ids if (trial_ids and len(trial_ids))
        else ([cond_alias.get(str(x).lower(), str(x).lower()) for x in condition_name]
              if condition_name is not None else [])
    )
    trial_end_abs = metrics.get("duration_end_abs", None) if isinstance(metrics, dict) else None

    save_res = save_onsets_offsets_by_condition(
        patient_id=patient_id, block_name=block_name,
        onsets=on_abs, offsets=off_abs, sampling_rate=sampling_rate,
        trial_ids=trial_ids_for_save,
        out_dir=os.path.join(save_path, "prep0"),
        include_invalid=True, cond_alias=cond_alias,
        trigger_label=str(trig_label), trial_end_abs=trial_end_abs,
        condition_name=condition_name, resp_accuracy=resp_accuracy,
        trial_idx=trial_idx_col,
    )
    print(f"  Saved: {save_res}")
