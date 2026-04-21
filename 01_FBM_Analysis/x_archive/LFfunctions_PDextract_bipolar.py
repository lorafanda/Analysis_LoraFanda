
import os
import json
import csv
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
import h5py
import scipy
import scipy.signal as sig
from scipy.io import loadmat
from scipy.signal import (
    savgol_filter as savitzky_golay,
    butter,
    lfilter,
    spectrogram,
    welch,
    iirnotch,
    filtfilt,
    sosfiltfilt,
    zpk2sos,
    tf2sos,
)
# Requires: numpy as np, matplotlib.pyplot as plt, pandas as pd, scipy.signal as sig

def onset_offset_by_peak_envelope(
    x1, x2=None, *, sampling_rate,
    t_start=0, t_end=None,
    bipolar_doflip=True, flip_trigs=False,
    # peak picking
    min_separation_s=0.25,
    prominence=None, prom_scale=1.0,
    # gates
    abs_amp_min=None, amp_scale=6.0,
    slope_min=None, slope_scale=3.0,
    min_event_separation_s=None,
    # behavior flags
    onoff_from_extrema=True,   # True => sign-of-extrema; False => zero-cross
    return_extrema=True,
    # --- plotting/meta like PD tool ---
    trial_ids=None,
    invalid_trials=None,
    ignore_invalid=False,
    plot_title=None,
    do_plot=False, figsize=(36, 4), connect_dots=True, save_plot_path=None,

    # --- NEW ---
    fake_trials=None,          # indices or IDs to DELETE entirely
    extra_table_path=None,     # CSV/TSV path in either schema you showed
):
    """
    Returns
    -------
    Standard:
      on_idx, off_idx, t_on_s, t_off_s, raw, t, summary
    If return_extrema=True, also:
      ext_idx_global, ext_t_s, ext_val
    If ignore_invalid=True and invalid_trials is not None, also:
      on_valid, off_valid
    """
    import numpy as np
    import scipy.signal as sig
    import matplotlib.pyplot as plt

    def _robust_scale(vec):
        vec = np.asarray(vec, float)
        med = np.nanmedian(vec); mad = np.nanmedian(np.abs(vec - med))
        return 1.4826 * mad if mad > 0 else (np.nanstd(vec) or 1.0)

    def _normalize_invalid_selector1(n_on, trial_ids, invalid_trials):
        # returns boolean mask length n_on; True=invalid
        m = np.zeros(n_on, dtype=bool)
        if invalid_trials is None or n_on == 0:
            return m
        arr = invalid_trials
        if not isinstance(arr, (list, tuple, set, np.ndarray)):
            arr = [arr]
        arr = np.asarray(list(arr))
        if arr.size == 0:
            return m
        if np.issubdtype(arr.dtype, np.integer):
            idx = np.clip(arr.astype(int), 0, max(0, n_on-1))
            m[idx] = True
            return m
        if trial_ids is not None:
            trid = np.asarray(trial_ids)
            try:
                arr_cast = arr.astype(trid.dtype, copy=False)
            except Exception:
                arr_cast = arr
            return np.isin(trid, arr_cast)
        # fallback try indices
        try:
            idx = np.clip(arr.astype(int), 0, max(0, n_on-1))
            m[idx] = True
        except Exception:
            pass
        return m

    def _normalize_selector_as_indices(on_list, trial_ids, pick):
        # returns mask over on_list (True = selected); or None if empty
        n = len(on_list)
        if pick is None or n == 0:
            return None
        if not isinstance(pick, (list, tuple, set, np.ndarray)):
            pick = [pick]
        arr = np.asarray(list(pick))
        if arr.size == 0:
            return None
        mask = np.zeros(n, dtype=bool)
        if np.issubdtype(arr.dtype, np.integer):
            idx = np.clip(arr.astype(int), 0, max(0, n-1))
            mask[idx] = True
            return mask
        if trial_ids is not None:
            trid = np.asarray(trial_ids)
            try:
                arr_cast = arr.astype(trid.dtype, copy=False)
            except Exception:
                arr_cast = arr
            return np.isin(trid, arr_cast)
        try:
            idx = np.clip(arr.astype(int), 0, max(0, n-1))
            mask[idx] = True
            return mask
        except Exception:
            return None

   
    def _read_trial_table(path):
        import pandas as pd
        import numpy as np

        # --- 1) Sniff delimiter from the first non-empty line
        sniff_sep = None
        with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Prefer ',' if present; else '\t' if present; else fallback
                    if ',' in line and '\t' not in line:
                        sniff_sep = ','
                    elif '\t' in line and ',' not in line:
                        sniff_sep = '\t'
                    elif ',' in line and '\t' in line:
                        # Mixed? prefer comma for your case
                        sniff_sep = ','
                    else:
                        sniff_sep = None
                    break

        # --- 2) Try to read with the sniffed sep first, then fallbacks
        def _try_read(sep):
            try:
                return pd.read_csv(
                    path,
                    sep=sep,
                    engine="python",
                    encoding="utf-8-sig",
                    skipinitialspace=True,
                    dtype=str,          # read as strings first, convert later
                )
            except Exception:
                return None

        df = None
        if sniff_sep is not None:
            df = _try_read(sniff_sep)
        if df is None:
            # fallback attempts
            for sep in (None, '\t', ','):
                df = _try_read(sep)
                if df is not None:
                    break
        if df is None:
            # last resort: raw read -> single column
            df = pd.read_csv(path, header=None, encoding="utf-8-sig", engine="python")

        # --- 3) If pandas still gave a single column that clearly contains commas, split it
        if df.shape[1] == 1:
            sole = df.columns[0]
            # If the column name itself contains commas, it's likely the header row got merged
            # Example: "TrialNumber,BlockNumber,..."
            if isinstance(sole, str) and ',' in sole:
                header_cols = [c.strip() for c in sole.split(',')]
                # Use the *values* column to split the data rows
                col0 = df.iloc[:, 0].astype(str)
                # Find the first row that actually looks like data (has commas)
                has_comma = col0.str.contains(',', regex=False)
                data_rows = col0[has_comma]
                # Split rows by comma
                split_vals = data_rows.str.split(',', expand=True)
                split_vals.columns = header_cols[:split_vals.shape[1]]
                df = split_vals
            else:
                # Maybe the *values* contain commas
                col0 = df.iloc[:, 0].astype(str)
                if col0.str.contains(',', regex=False).any():
                    split_vals = col0.str.split(',', expand=True)
                    # Promote first row to header if it looks like header strings
                    maybe_header = split_vals.iloc[0].tolist()
                    if all(isinstance(x, str) for x in maybe_header):
                        split_vals.columns = [str(x).strip() for x in maybe_header]
                        split_vals = split_vals.iloc[1:].reset_index(drop=True)
                    df = split_vals

        # Clean up: drop all-empty columns/rows
        df = df.dropna(how="all")
        # Strip whitespace from header
        df.columns = [str(c).strip() for c in df.columns]

        # Build lowercase alias view without losing original df for preview
        cols_lower = {c: c.lower() for c in df.columns}
        dfl = df.rename(columns=cols_lower)

        # ---- Extract duration: prefer 'duration', fallback 'responsetime'/'response_time'
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

        # ---- Explicit onset column?
        if "onset" in dfl.columns:
            onset_sec = pd.to_numeric(dfl["onset"], errors="coerce").to_numpy()
            return {
                'mode': 'by_onset',
                'onset_sec': onset_sec,
                'duration_sec': duration_sec,
                'trial_ids': None,
                'raw_df': df,
                'source_path': path,
                'duration_col_used': dur_col
            }

        # ---- Decide by first column type
        if df.shape[1] > 0:
            first = df.columns[0]
            s0 = pd.to_numeric(df[first], errors="coerce")
            if s0.notna().all():
                if np.all(np.equal(np.mod(s0, 1), 0)):  # all ints => trial index file
                    return {
                        'mode': 'by_index',
                        'onset_sec': None,
                        'duration_sec': duration_sec,
                        'trial_ids': s0.astype(int).to_numpy(),
                        'raw_df': df,
                        'source_path': path,
                        'duration_col_used': dur_col
                    }
                else:
                    return {
                        'mode': 'by_onset',
                        'onset_sec': s0.to_numpy(float),
                        'duration_sec': duration_sec,
                        'trial_ids': None,
                        'raw_df': df,
                        'source_path': path,
                        'duration_col_used': dur_col
                    }

        # ---- Fallback: search for a trial column by name
        trial_col = None
        for c in df.columns:
            if c.lower() in ("trialnumber","trial","trial_num","trial_index"):
                trial_col = c; break
        if trial_col is not None:
            trid = pd.to_numeric(df[trial_col], errors="coerce").fillna(-1).astype(int).to_numpy()
            return {
                'mode': 'by_index',
                'onset_sec': None,
                'duration_sec': duration_sec,
                'trial_ids': trid,
                'raw_df': df,
                'source_path': path,
                'duration_col_used': dur_col
            }

        # ---- Last resort: treat as by_index with no IDs
        return {
            'mode': 'by_index',
            'onset_sec': None,
            'duration_sec': duration_sec,
            'trial_ids': None,
            'raw_df': df,
            'source_path': path,
            'duration_col_used': dur_col
        }




    fs = float(sampling_rate)
    lockout_samp = 0 if not min_event_separation_s else int(round(min_event_separation_s * fs))

    # ----- montage & slice -----
    x1 = np.asarray(x1, float)
    if x2 is None:
        d = x1
    else:
        x2 = np.asarray(x2, float)
        d = (x1 - x2) if bipolar_doflip else (x2 - x1)

    if t_end is None or t_end < 0:
        raw = d[int(t_start):]; base_shift = int(t_start)
    else:
        raw = d[int(t_start):int(t_end)]; base_shift = int(t_start)
    if flip_trigs:
        raw = -raw

    N = len(raw); t = np.arange(N, dtype=float) / fs

    # ----- thresholds -----
    sigma = _robust_scale(raw)
    amp_thresh   = abs_amp_min if (abs_amp_min is not None) else (amp_scale * sigma)
    slope_thresh = slope_min    if (slope_min is not None)    else (slope_scale * sigma)
    if prominence is None:
        prominence = prom_scale * sigma
    dist = max(1, int(round(min_separation_s * fs)))

    # ----- extrema -----
    pos_idx, _ = sig.find_peaks(raw,  prominence=prominence, distance=dist)
    neg_idx, _ = sig.find_peaks(-raw, prominence=prominence, distance=dist)
    if pos_idx.size == 0 and neg_idx.size == 0:
        ext_idx = np.array([], dtype=int)
        ext_val = np.array([], dtype=float)
    else:
        ext_idx = np.concatenate([pos_idx, neg_idx])
        ext_val = np.concatenate([raw[pos_idx], raw[neg_idx]])
        order = np.argsort(ext_idx)
        ext_idx = ext_idx[order]; ext_val = ext_val[order]
        # compress adjacent same-sign extrema (keep stronger |value|)
        if ext_idx.size:
            keep = [0]
            for k in range(1, len(ext_idx)):
                if np.sign(ext_val[k]) == np.sign(ext_val[keep[-1]]):
                    if abs(ext_val[k]) > abs(ext_val[keep[-1]]):
                        keep[-1] = k
                else:
                    keep.append(k)
            ext_idx = ext_idx[keep]; ext_val = ext_val[keep]

    ext_idx_global = ext_idx + base_shift
    ext_t_s = ext_idx / fs

    # ----- build onsets/offsets (unchanged logic) -----
    on_idx, off_idx, t_on_s, t_off_s = [], [], [], []
    last_on_global = last_off_global = -10**18

    if onoff_from_extrema:
        for i, a in zip(ext_idx, ext_val):
            if abs(a) < amp_thresh:
                continue
            gz = int(i) + base_shift
            tz = (i) / fs
            if a > 0:
                if (gz - last_on_global) >= lockout_samp:
                    on_idx.append(gz); t_on_s.append(tz); last_on_global = gz
            elif a < 0:
                if (gz - last_off_global) >= lockout_samp:
                    off_idx.append(gz); t_off_s.append(tz); last_off_global = gz
        method = "extrema_sign"
    else:
        for k in range(1, len(ext_idx)):
            a1, a2 = ext_val[k-1], ext_val[k]
            if a1 == 0 or a2 == 0 or np.sign(a1) == np.sign(a2):
                continue
            i1, i2 = int(ext_idx[k-1]), int(ext_idx[k])
            if max(abs(a1), abs(a2)) < amp_thresh:
                continue
            slope = abs(a2 - a1) / max(1, (i2 - i1))
            if slope < slope_thresh:
                continue
            alpha = (-a1) / (a2 - a1)
            iz = i1 + alpha * (i2 - i1)
            gz = int(round(iz)) + base_shift
            tz = iz / fs
            if a1 < 0 < a2:
                if (gz - last_on_global) >= lockout_samp:
                    on_idx.append(gz);  t_on_s.append(tz); last_on_global = gz
            elif a1 > 0 > a2:
                if (gz - last_off_global) >= lockout_samp:
                    off_idx.append(gz); t_off_s.append(tz); last_off_global = gz
        method = "zero_cross"

    on_idx  = np.array(on_idx,  dtype=int)
    off_idx = np.array(off_idx, dtype=int)
    t_on_s  = np.array(t_on_s,  dtype=float)
    t_off_s = np.array(t_off_s, dtype=float)

    # ---------- Remove FAKE trials first ----------
    fake_mask = _normalize_selector_as_indices(on_idx, trial_ids, fake_trials)
    if fake_mask is not None:
        keep = ~fake_mask
        on_idx = on_idx[keep]
        t_on_s = t_on_s[keep]
        # keep pairing if possible
        if len(off_idx) == len(fake_mask):
            off_idx = off_idx[keep]
            t_off_s = t_off_s[keep]
        # shrink trial_ids if provided
        if trial_ids is not None:
            trial_ids = np.asarray(trial_ids)[keep]

    summary = dict(
        method=method,
        n_onsets=int(len(on_idx)), n_offsets=int(len(off_idx)),
        prominence=float(prominence), min_separation_s=float(min_separation_s),
        amp_thresh=float(amp_thresh), slope_thresh=float(slope_thresh),
        min_event_separation_s=float(min_event_separation_s or 0.0),
        fs=fs, sigma=float(sigma), N=int(N),
        extrema_n=int(len(ext_idx)),
    )

    # ----- plotting (parity with PD tool + ORANGE overlays) -----
    if do_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        if N:
            ax.plot(t, raw, lw=0.9, alpha=0.9, label="raw (X1−X2)")
        if connect_dots and len(ext_idx):
            ax.plot(ext_t_s, ext_val, "-o", ms=3, label="extrema envelope")

        n = len(on_idx)
        is_invalid = _normalize_invalid_selector1(n, trial_ids, invalid_trials)

        # y limits and headroom
        y_min = np.nanmin(raw) if N else -1.0
        y_max = np.nanmax(raw) if N else 1.0
        y_span = (y_max - y_min) if (y_max > y_min) else 1.0
        y_top = y_max + 0.05 * y_span

        # draw PD-like onsets/offsets + labels + shading
        for k in range(n):
            onset_samp_abs = on_idx[k]
            onset_t = (onset_samp_abs - base_shift) / fs
            if onset_t < 0 or onset_t > (N / fs):
                continue

            invalid = bool(is_invalid[k])
            vcolor = "red" if invalid else "tab:green"
            shade_color = (1.0, 0.8, 0.8, 0.25) if invalid else (0.8, 1.0, 0.8, 0.22)

            ax.axvline(onset_t, ymin=0, ymax=1, linestyle="--", linewidth=1.25, color=vcolor, alpha=0.9)

            if trial_ids is not None and k < len(trial_ids):
                label_txt = f"{k}:{trial_ids[k]}"
            else:
                label_txt = f"{k}"
            ax.text(onset_t, y_top, label_txt,
                    ha="center", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=vcolor, lw=1, alpha=0.9))

            # optional offset line & shading
            if k < len(off_idx):
                off_samp_abs = off_idx[k]
                off_t = (off_samp_abs - base_shift) / fs
                if 0.0 <= off_t <= (N / fs):
                    ax.axvline(off_t, ymin=0, ymax=1, linestyle=":", linewidth=1.25, color=vcolor, alpha=0.9)
                left = max(onset_t, 0.0)
                right = min(off_t, (N / fs)) if off_samp_abs > onset_samp_abs else left
                if right > left:
                    ax.axvspan(left, right, color=shade_color, linewidth=0)

        # ----- ORANGE overlays from extra table -----
        orange_on_abs = None
        orange_end_abs = None
        if extra_table_path is not None and len(on_idx) > 0:
            extra = _read_trial_table(extra_table_path)
            if extra['mode'] == 'by_index':
                # No onsets in file; just durations mapped to detections
                dur_sec = extra['duration_sec']
                if dur_sec is not None:
                    nmap = min(len(dur_sec), len(on_idx))
                    orange_end_abs = on_idx[:nmap] + np.round(dur_sec[:nmap] * fs).astype(int)
            elif extra['mode'] == 'by_onset':
                onset_sec = extra['onset_sec']
                if onset_sec is not None and len(onset_sec) > 0:
                    # align file's first onset to first detection
                    shift_samp = int(on_idx[0] - round(onset_sec[0] * fs))
                    orange_on_abs = np.round(onset_sec * fs).astype(int) + shift_samp
                    dur_sec = extra['duration_sec']
                    if dur_sec is not None and len(dur_sec) == len(onset_sec):
                        orange_end_abs = orange_on_abs + np.round(dur_sec * fs).astype(int)

        # draw orange onset (shifted) and orange end
        if orange_on_abs is not None:
            for s in orange_on_abs:
                rel_t = (s - base_shift) / fs
                if 0.0 <= rel_t <= (N / fs):
                    ax.axvline(rel_t, ymin=0, ymax=1, linestyle="-.", linewidth=1.25, color="orange", alpha=0.95)
        if orange_end_abs is not None:
            for s in orange_end_abs:
                rel_t = (s - base_shift) / fs
                if 0.0 <= rel_t <= (N / fs):
                    ax.axvline(rel_t, ymin=0, ymax=1, linestyle="-", linewidth=1.25, color="orange", alpha=0.95)

        # legend proxy
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], color="tab:green", lw=2, linestyle="--", label="Valid onset"),
            Line2D([0], [0], color="red", lw=2, linestyle="--", label="Invalid onset"),
            Line2D([0], [0], color="k", lw=1, linestyle="-", label="Raw / envelope"),
        ]
        if orange_on_abs is not None:
            legend_elems.append(Line2D([0], [0], color="orange", lw=2, linestyle="-.", label="File onset (shifted)"))
        if orange_end_abs is not None:
            legend_elems.append(Line2D([0], [0], color="orange", lw=2, linestyle="-", label="File end (onset+duration)"))
        ax.legend(handles=legend_elems, loc="upper right", framealpha=0.9)

        # axes cosmetics
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        title = plot_title or f"{method} | onsets={len(on_idx)} offsets={len(off_idx)}"
        title += f" | amp≥{amp_thresh:.3g}"
        if not onoff_from_extrema:
            title += f", slope≥{slope_thresh:.3g}/samp"
        ax.set_title(title)
        ax.set_ylim(y_min - 0.05*y_span, y_top + 0.10*y_span)
        ax.grid(alpha=.25)
        fig.tight_layout()
        if save_plot_path:
            fig.savefig(save_plot_path, dpi=150)
            plt.close(fig)
        else:
            try:
                plt.show()
            finally:
                plt.close(fig)

    # ----- returns (backward compatible) -----
    base_returns = (on_idx, off_idx, t_on_s, t_off_s, raw, t, summary)
    if return_extrema:
        base_returns = base_returns + (ext_idx_global, ext_t_s, ext_val)

    if ignore_invalid and invalid_trials is not None and len(on_idx):
        mask = ~_normalize_invalid_selector1(len(on_idx), trial_ids, invalid_trials)
        on_valid = on_idx[mask]
        if len(off_idx) == len(on_idx):
            off_valid = off_idx[mask]
        else:
            off_valid = off_idx[:mask.sum()]
        return base_returns + (on_valid, off_valid)

    return base_returns
