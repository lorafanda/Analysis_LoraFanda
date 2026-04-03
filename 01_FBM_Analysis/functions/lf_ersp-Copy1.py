# lf_ersp.py
from __future__ import annotations
from dataclasses import dataclass
import os, numpy as np
from fractions import Fraction
from scipy.signal import spectrogram, resample_poly
import matplotlib.pyplot as plt

@dataclass
class ERSPParams:
    nperseg: int = 128
    noverlap: int = 96*2
    nfft: int = 1024
    baseline_w: tuple[float,float] = (-0.4, 0)
    baseline_calc_w: tuple[float,float] = (-0.4, -0.1)
    proportions: tuple[float,float,float] = (0, 0.50, 0.50)  # base/stim/post
    n_time_bins: int = 300
    crossfade_pct: float = 0.0
    pad_pct: float = 0.0
    pad_mode: str = "reflect"
    vmin: float = -8.0
    vmax: float =  8.0
    fmax: float = 400.0

def _to_khz_resampled(x: np.ndarray, fs_in: float):
    frac = Fraction((1000.0 / fs_in)).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    y = x if (up == 1 and down == 1) else resample_poly(x, up=up, down=down)
    return y, fs_in * (up / down), (up / down)

def _spectro(seg: np.ndarray, fs: float, p: ERSPParams):
    f, t, Sxx = spectrogram(seg, fs=fs, nperseg=p.nperseg, noverlap=p.noverlap, nfft=p.nfft,
                            scaling="density", mode="psd")
    Sxx = np.maximum(Sxx, 1e-20)
    return f, t, 10.0*np.log10(Sxx)

def _pad_and_spectro(seg: np.ndarray, fs: float, p: ERSPParams, align_sec: float):
    L = len(seg)
    if L <= 1:
        f, t_rel, S = _spectro(seg, fs, p)
        return f, align_sec + t_rel, S
    padN = int(round(max(0.0, p.pad_pct)*L))
    seg_pad = np.pad(seg, (padN, padN), mode=p.pad_mode) if padN > 0 else seg
    f, t_pad, S_pad = _spectro(seg_pad, fs, p)
    pad_sec = padN / fs
    t_abs = (t_pad - pad_sec) + align_sec
    t0, t1 = align_sec, align_sec + (L / fs)
    keep = (t_abs >= t0) & (t_abs < t1)
    if not np.any(keep):
        return f, t_abs, S_pad
    return f, t_abs[keep], S_pad[:, keep]

def compute_ersp(
    signals: np.ndarray,  # (n_samples, n_channels)
    fs: float,
    onsets: np.ndarray,
    offsets: np.ndarray | None,
    channel_idx: int,
    *,
    trial_ends: np.ndarray | None = None,
    mode: str = "TN",                      # "RT" or "TN"
    time_window: tuple[float,float] = (-1.0, 3.0),  # RT only
    params: ERSPParams = ERSPParams(),
) -> dict:
    assert signals.ndim == 2, "signals must be 2D"
    sig_in = signals[:, int(channel_idx)].astype(float)
    sig_ds, fs_ds, scale = _to_khz_resampled(sig_in, float(fs))
    on_ds  = np.round(np.asarray(onsets,  float)*scale).astype(int)
    off_ds = np.round(np.asarray(offsets, float)*scale).astype(int) if (offsets is not None and len(offsets)) else np.array([], int)
    te_ds  = np.round(np.asarray(trial_ends, float)*scale).astype(int) if trial_ends is not None else None

    base_w = params.baseline_w
    calc_w = params.baseline_w if params.baseline_calc_w is None else params.baseline_calc_w

    if str(mode).upper() == "RT":
        trials_db, trials_z = [], []
        for on in on_ds:
            s = int(on + time_window[0]*fs_ds)
            e = int(on + time_window[1]*fs_ds)
            s = max(0, s); e = min(len(sig_ds), max(s+1, e))
            seg = sig_ds[s:e]
            f, t_sec, S_db = _pad_and_spectro(seg, fs_ds, params, time_window[0])
            bmask = (t_sec >= calc_w[0]) & (t_sec < calc_w[1])
            if not np.any(bmask):
                nT = S_db.shape[1]; bmask = np.zeros(nT, bool); bmask[:max(2, nT//10)] = True
            mu = np.mean(S_db[:, bmask], axis=1, keepdims=True)
            sd = np.std (S_db[:, bmask], axis=1, keepdims=True, ddof=1)
            trials_db.append((t_sec, S_db - mu))
            trials_z .append((t_sec, (S_db - mu)/(sd + 1e-12)))

        max_cols = max(A.shape[1] for _, A in trials_db) if trials_db else 0
        if max_cols == 0:
            raise ValueError("No valid trials to average.")
        x = np.linspace(time_window[0], time_window[1], max_cols, endpoint=False)

        def _regrid(trials):
            mats, f_ref = [], None
            for t_i, A_i in trials:
                if f_ref is None: f_ref = f
                out = np.full((A_i.shape[0], x.size), np.nan)
                t_u, idx = np.unique(t_i, return_index=True)
                if t_u.size >= 2 and t_u[-1] > t_u[0]:
                    A_u = A_i[:, idx]
                    for r in range(A_u.shape[0]):
                        out[r, :] = np.interp(x, t_u, A_u[r, :], left=np.nan, right=np.nan)
                mats.append(out)
            with np.errstate(invalid="ignore"): avg = np.nanmean(np.stack(mats, 0), axis=0)
            return f_ref, avg

        f_out, avg_db = _regrid(trials_db)
        _,     avg_z  = _regrid(trials_z)

        if offsets is not None and len(offsets) == len(onsets) and len(onsets) > 0:
            offset_marker = float(np.mean((np.asarray(offsets) - np.asarray(onsets)) / float(fs)))
        else:
            offset_marker = (time_window[1] - time_window[0]) * 0.5

        return dict(
            avg_db=avg_db, avg_z=avg_z, f=f_out, x=x,
            markers=dict(onset=0.0, offset=offset_marker),
            meta=dict(mode="RT", fs_in=float(fs), fs_ds=float(fs_ds), scale=float(scale))
        )

    # ---- TN (piecewise-normalized) ----
    pB, pS, pP = params.proportions
    tot = pB + pS + pP
    if not np.isclose(tot, 1.0): pB, pS, pP = (pB/tot, pS/tot, pP/tot)
    nB = max(2, int(round(pB * params.n_time_bins)))
    nS = max(2, int(round(pS * params.n_time_bins)))
    nP = max(2, int(round(pP * params.n_time_bins)))
    nP += (params.n_time_bins - (nB + nS + nP))
    i0, i1, i2, i3 = 0, nB, nB + nS, params.n_time_bins
    x = np.linspace(0.0, 100.0, params.n_time_bins, endpoint=True)

    warped_db, warped_z = [], []
    f_common = None

    for i in range(len(on_ds)):
        on = on_ds[i]
        s = on + int(round(base_w[0]*fs_ds))
        e = int(te_ds[i])
        s = max(0, s); e = min(len(sig_ds), max(s+1, e))
        seg = sig_ds[s:e]

        f, t_rel, S_db = _pad_and_spectro(seg, fs_ds, params, align_sec=base_w[0])
        bmask = (t_rel >= (calc_w[0])) & (t_rel < (calc_w[1] if params.baseline_calc_w else 0.0))
        if not np.any(bmask):
            nT = S_db.shape[1]; bmask = np.zeros(nT, bool); bmask[:max(2, nT//10)] = True
        mu = np.mean(S_db[:, bmask], axis=1, keepdims=True)
        sd = np.std (S_db[:, bmask], axis=1, keepdims=True, ddof=1)
        Srel = S_db - mu
        Zrel = (S_db - mu)/(sd + 1e-12)

        off_t = ((off_ds[i] - on_ds[i]) / fs_ds) if i < len(off_ds) else np.nan
        if not np.isfinite(off_t) or off_t <= 0:
            valid = (off_ds[:min(len(off_ds), len(on_ds))] - on_ds[:min(len(off_ds), len(on_ds))]) / fs_ds
            valid = valid[np.isfinite(valid) & (valid > 0)]
            off_t = float(np.median(valid)) if valid.size else 0.5
        next_on_t = ((te_ds[i] - on_ds[i]) / fs_ds) 

        eps = 1.0 / fs_ds
        off_t = max(eps, float(off_t))
        next_on_t = max(off_t + eps, float(next_on_t))

        mask_base = (t_rel >= base_w[0]) & (t_rel < 0.0)
        mask_stim = (t_rel >= 0.0) & (t_rel < off_t)
        mask_post = (t_rel >= off_t) & (t_rel <= next_on_t)

        Wdb = np.full((Srel.shape[0], params.n_time_bins), np.nan); Wz = Wdb.copy()

        def _warp_piece(t_src, A_src, i_start, i_end, a0, a1, W):
            if i_end <= i_start or t_src.size < 2: return
            t_u, idx = np.unique(t_src, return_index=True)
            if t_u.size < 2 or t_u[-1] <= t_u[0]: return
            A_u = A_src[:, idx]
            u = np.linspace(a0, a1, i_end - i_start, endpoint=False)
            for r in range(A_u.shape[0]):
                W[r, i_start:i_end] = np.interp(u, t_u, A_u[r, :], left=np.nan, right=np.nan)

        _warp_piece(t_rel[mask_base], Srel[:, mask_base], i0, i1, base_w[0], 0.0, Wdb)
        _warp_piece(t_rel[mask_stim], Srel[:, mask_stim], i1, i2, 0.0, off_t,    Wdb)
        _warp_piece(t_rel[mask_post], Srel[:, mask_post], i2, i3, off_t, next_on_t, Wdb)

        _warp_piece(t_rel[mask_base], Zrel[:, mask_base], i0, i1, base_w[0], 0.0, Wz)
        _warp_piece(t_rel[mask_stim], Zrel[:, mask_stim], i1, i2, 0.0, off_t,    Wz)
        _warp_piece(t_rel[mask_post], Zrel[:, mask_post], i2, i3, off_t, next_on_t, Wz)

        warped_db.append(Wdb); warped_z.append(Wz)
        if f_common is None: f_common = f

    warped_db = np.stack(warped_db, 0); warped_z = np.stack(warped_z, 0)
    with np.errstate(invalid="ignore"):
        avg_db = np.nanmean(warped_db, 0)
        avg_z  = np.nanmean(warped_z,  0)

    # cosmetic crossfade at boundaries (dB only)
    w = max(1, int(round(params.crossfade_pct * params.n_time_bins)))
    def _blend(col_left, col_right, center, M):
        L = M[:, max(col_left, 0)]
        R = M[:, min(col_right, M.shape[1]-1)]
        for k, c in enumerate(range(center - w, center + w)):
            if 0 <= c < M.shape[1]:
                alpha = (k + 1) / (2 * w)
                M[:, c] = (1 - alpha) * L + alpha * R
    _blend(i1-1, i1, i1, avg_db)
    _blend(i2-1, i2, i2, avg_db)

    return dict(
        avg_db=avg_db, avg_z=avg_z, f=f_common, x=x,
        markers=dict(onset=100.0*(pB), offset=100.0*(pB+pS)),
        meta=dict(mode="TN", fs_in=float(fs), fs_ds=float(fs_ds), scale=float(scale),
                  proportions=(pB,pS,pP), bins=(nB,nS,nP))
    )

def plot_ersp(
    ersp: dict,
    *,
    patient_id: str,
    condition: str,
    reref_type: str,
    chan_name: str,
    save_dir: str | None = None,
    params: ERSPParams = ERSPParams(),
) -> str:
    f = ersp["f"]; x = ersp["x"]; A = ersp["avg_db"]
    mode = ersp["meta"]["mode"]; mk = ersp["markers"]

    plt.figure(figsize=(8,5))
    plt.pcolormesh(x, f, A, shading="gouraud", cmap="bwr", vmin=params.vmin, vmax=params.vmax)
    cb = plt.colorbar(); cb.set_label("Power change (dB)")
    title = f"{patient_id} – {condition} – {reref_type}-ref ERSP: {chan_name}"
    if mode == "TN":
        pB,pS,pP = ersp["meta"]["proportions"]
        title += f" ({int(round(100*pB))}/{int(round(100*pS))}/{int(round(100*pP))}% time-normalized)"
    plt.title(title); plt.ylabel("Frequency (Hz)"); plt.ylim(0, params.fmax)

    if mode == "RT":
        plt.xlabel("Time (s) relative to onset)")
        plt.axvline(0.0, color="k", ls="-", lw=1.5, label="stim")
        if mk.get("offset") is not None:
            plt.axvline(mk["offset"], color="k", ls="--", lw=1.5, label="resp")
    else:
        plt.xlabel("Normalized time (%)")
        ticks = [0.0, mk["onset"], mk["offset"], 100.0]
        plt.xticks(ticks, [f"{t:.0f}" for t in ticks])
        plt.axvline(mk["onset"],  color="k", ls="-",  lw=1.5, label=f"Stim Onset ({mk['onset']:.0f}%)")
        plt.axvline(mk["offset"], color="k", ls="--", lw=1.5, label=f"Response Onset ({mk['offset']:.0f}%)")
    plt.legend(loc="upper right"); plt.tight_layout()

    fname = f"{patient_id}_{condition}_{reref_type}_ERSP_{chan_name}{('_TN' if mode=='TN' else '')}.tif"
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, fname)
    plt.savefig(fname, dpi=300, transparent=True, format="tiff")
    plt.close()

    # sidecar
    sidecar = fname.replace(".tif", "_metrics.npz")
    try:
        np.savez_compressed(sidecar, avg_db=ersp["avg_db"], avg_z=ersp["avg_z"], f=f, x=x, markers=mk, meta=ersp["meta"])
    except Exception:
        pass
    return fname
