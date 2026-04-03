# lf_ersp.py
from __future__ import annotations
from dataclasses import dataclass
import os, numpy as np
from fractions import Fraction
from scipy.signal import spectrogram, resample_poly, butter, filtfilt, hilbert
import matplotlib.pyplot as plt
from scipy.signal.windows import dpss
from numpy.fft import rfft
import numpy as np
from scipy.ndimage import gaussian_filter, label, generate_binary_structure

# ----------------------------
# Parameters
# ----------------------------
@dataclass
class ERSPParams:
    nperseg: int = 128
    noverlap: int = 96                 # 75% overlap (< nperseg)
    nfft: int = 1024
    baseline_w: tuple[float,float] = (-0.4, 0.0)
    baseline_calc_w: tuple[float,float] | None = (-0.4, -0.1)
    proportions: tuple[float,float,float] = (0.0, 0.50, 0.50)  # base/stim/post
    n_time_bins: int = 400
    crossfade_pct: float = 0.0         # cosmetic seam blending on avg_db only
    pad_pct: float = 0.1
    pad_mode: str = "reflect"
    vmin: float = -7.0
    vmax: float =  7.0
    fmax: float = 400.0
    
    # Multitaper controls
    use_multitaper = False
    mt_NW = 2
    mt_Kmax = 5

# --------------------------------------------------
# Utility helpers
# --------------------------------------------------

def _to_khz_resampled(x: np.ndarray, fs_in: float):
    frac = Fraction((1000.0 / fs_in)).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    y = x if (up == 1 and down == 1) else resample_poly(x, up=up, down=down)
    return y, fs_in * (up / down), (up / down)

def _centers_to_edges(c: np.ndarray) -> np.ndarray:
    """Convert center coordinates to bin edges for pcolormesh."""
    c = np.asarray(c, float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5], float)
    step = np.diff(c)
    # internal edges are midpoints; pad first/last by half-step
    return np.concatenate((
        [c[0] - step[0]/2.0],
        (c[:-1] + c[1:]) / 2.0,
        [c[-1] + step[-1]/2.0]
    ))

# --------------------------------------------------
# Spectrogram backends
# --------------------------------------------------

def _spectro(seg: np.ndarray, fs: float, p):
    """
    Standard Hann-window spectrogram.
    """
    nfft = p.nfft if p.nfft is not None else p.nperseg
    nfft = max(p.nperseg, min(nfft, 2*p.nperseg))  # cap to avoid huge zero-padding

    f, t, Sxx = spectrogram(
        seg, fs=fs,
        window='hann',  # make taper explicit
        nperseg=p.nperseg,
        noverlap=p.noverlap,
        nfft=nfft,
        detrend=False,
        scaling="density",
        mode="psd"
    )
    Sxx = np.maximum(Sxx, 1e-20)
    return f, t, 10.0*np.log10(Sxx)


def _spectro_multitaper(seg: np.ndarray, fs: float, p):
    """
    Multitaper spectrogram using DPSS tapers.
    """
    nperseg = p.nperseg
    noverlap = p.noverlap
    nfft = max(nperseg, min(p.nfft, 2*nperseg))

    step = nperseg - noverlap
    nwin = 1 + (len(seg) - nperseg) // step if len(seg) >= nperseg else 0

    # Time-bandwidth product (NW) controls taper width
    NW = getattr(p, "mt_NW", 3)          # default 3
    Kmax = getattr(p, "mt_Kmax", 2*NW)   # default 2*NW tapers
    tapers, eigs = dpss(nperseg, NW, Kmax, return_ratios=True)

    freqs = np.fft.rfftfreq(nfft, 1/fs)
    times = []
    Sxx = np.zeros((len(freqs), nwin))

    for i in range(nwin):
        start = i * step
        end = start + nperseg
        xseg = seg[start:end]
        if len(xseg) < nperseg:
            break

        Sk = []
        for k in range(Kmax):
            x_tapered = xseg * tapers[k]
            Xf = rfft(x_tapered, n=nfft)
            Pxx = (np.abs(Xf)**2) / (fs * np.sum(tapers[k]**2))
            Sk.append(Pxx)
        Sxx[:, i] = np.mean(Sk, axis=0)
        times.append((start + nperseg/2) / fs)  # center of window

    Sxx = np.maximum(Sxx, 1e-20)
    return freqs, np.array(times), 10*np.log10(Sxx)

# --------------------------------------------------
# Wrapper with optional padding
# --------------------------------------------------

def _pad_and_spectro(seg: np.ndarray, fs: float, p, align_sec: float):
    L = len(seg)
    if L <= 1:
        f, t_rel, S = _spectro(seg, fs, p)
        return f, align_sec + t_rel, S

    # Optional padding
    padN = 0
    if getattr(p, 'pad_pct', 0) and p.pad_pct > 0:
        padN = min(int(round(p.pad_pct * L)), p.nperseg // 2)
    seg_pad = np.pad(seg, (padN, padN), mode=getattr(p, 'pad_mode', 'reflect')) if padN > 0 else seg
    pad_sec = padN / fs

    # Choose backend: multitaper or Hann
    if getattr(p, "use_multitaper", False):
        f, t_pad, S_pad = _spectro_multitaper(seg_pad, fs, p)
    else:
        f, t_pad, S_pad = _spectro(seg_pad, fs, p)

    t_abs = (t_pad - pad_sec) + align_sec
    t0, t1 = align_sec, align_sec + (L / fs)
    keep = (t_abs >= t0) & (t_abs < t1)
    if not np.any(keep):
        return f, t_abs, S_pad
    return f, t_abs[keep], S_pad[:, keep]


# ----------------------------
# ERSP (Avg over trials)
# ----------------------------
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

    # ---- RT (fixed real time window) ----
    if str(mode).upper() == "RT":
        trials_db, trials_z = [], []
        f_first = None
        for on in on_ds:
            s = int(on + time_window[0]*fs_ds)
            e = int(on + time_window[1]*fs_ds)
            s = max(0, s); e = min(len(sig_ds), max(s+1, e))
            seg = sig_ds[s:e]
            f, t_sec, S_db = _pad_and_spectro(seg, fs_ds, params, time_window[0])
            if f_first is None: f_first = f
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
            mats = []
            for t_i, A_i in trials:
                out = np.full((A_i.shape[0], x.size), np.nan)
                t_u, idx = np.unique(t_i, return_index=True)
                if t_u.size >= 2 and t_u[-1] > t_u[0]:
                    A_u = A_i[:, idx]
                    for r in range(A_u.shape[0]):
                        out[r, :] = np.interp(x, t_u, A_u[r, :], left=np.nan, right=np.nan)
                mats.append(out)
            with np.errstate(invalid="ignore"):
                avg = np.nanmean(np.stack(mats, 0), axis=0)
            return avg

        avg_db = _regrid(trials_db)
        avg_z  = _regrid(trials_z)

        if offsets is not None and len(offsets) == len(onsets) and len(onsets) > 0:
            offset_marker = float(np.mean((np.asarray(offsets) - np.asarray(onsets)) / float(fs)))
        else:
            offset_marker = (time_window[1] - time_window[0]) * 0.5

        return dict(
            avg_db=avg_db, avg_z=avg_z, f=f_first, x=x,
            markers=dict(onset=0.0, offset=offset_marker),
            meta=dict(mode="RT", fs_in=float(fs), fs_ds=float(fs_ds), scale=float(scale))
        )

    # ---- TN (piecewise-normalized) ----
    pB, pS, pP = params.proportions
    tot = pB + pS + pP
    if not np.isclose(tot, 1.0): pB, pS, pP = (pB/tot, pS/tot, pP/tot)
    N = params.n_time_bins

    def _alloc_bins(prop, N):
        if prop <= 0: return 0
        return max(2, int(round(prop * N)))

    nB = _alloc_bins(pB, N)
    nS = _alloc_bins(pS, N)
    nP = _alloc_bins(pP, N)
    nP += (N - (nB + nS + nP))  # fix rounding remainder

    i0, i1, i2, i3 = 0, nB, nB + nS, params.n_time_bins
    x = np.linspace(0.0, 100.0, params.n_time_bins, endpoint=False)

    warped_db, warped_z = [], []
    f_common = None

    for i in range(len(on_ds)):
        on = on_ds[i]
        s = on + int(round(base_w[0]*fs_ds))
        e = int(te_ds[i])
        s = max(0, s); e = min(len(sig_ds), max(s+1, e))
        seg = sig_ds[s:e]

        f, t_rel, S_db = _pad_and_spectro(seg, fs_ds, params, align_sec=base_w[0])
        if f_common is None: f_common = f

        # baseline for stats
        calc_w = params.baseline_w if params.baseline_calc_w is None else params.baseline_calc_w
        bmask = (t_rel >= calc_w[0]) & (t_rel < calc_w[1])
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

    warped_db = np.stack(warped_db, 0); warped_z = np.stack(warped_z, 0)
    with np.errstate(invalid="ignore"):
        avg_db = np.nanmean(warped_db, 0)
        avg_z  = np.nanmean(warped_z,  0)

    # cosmetic crossfade at boundaries (dB only) — NaN-aware half-cosine
    w = max(1, int(round(params.crossfade_pct * params.n_time_bins)))
    if w > 0:
        def _blend_nanaware(col_left, col_right, center, M, w_):
            L = M[:, max(col_left, 0)]
            R = M[:, min(col_right, M.shape[1]-1)]
            for k, c in enumerate(range(center - w_, center + w_)):
                if 0 <= c < M.shape[1]:
                    # half-cosine easing 0→1 over 2w steps
                    alpha = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / (2 * w_))
                    aL = np.where(np.isfinite(L), 1.0 - alpha, 0.0)
                    aR = np.where(np.isfinite(R), alpha,        0.0)
                    denom = aL + aR
                    out = np.divide(aL * L + aR * R, denom, out=np.zeros_like(L), where=denom > 0)
                    M[:, c] = out

        _blend_nanaware(i1-1, i1, i1, avg_db, w)
        _blend_nanaware(i2-1, i2, i2, avg_db, w)
    
    
    print(f"[ERSP TN] matrix size: time_bins={x.size}, freq_bins={f_common.size}  ->  {x.size}×{f_common.size}")

    return dict(
        avg_db=avg_db, avg_z=avg_z, f=f_common, x=x,
        markers=dict(onset=100.0*(pB), offset=100.0*(pB+pS)),
        meta=dict(mode="TN", fs_in=float(fs), fs_ds=float(fs_ds), scale=float(scale),
                  proportions=(pB,pS,pP), bins=(nB,nS,nP))
    )


def smooth_and_mask_ersp(
    A_db: np.ndarray,          # [F x T] (e.g., ersp["avg_db"])
    f: np.ndarray,             # freq centers (Hz)
    t: np.ndarray,             # time centers (seconds) OR normalized % (still fine)
    *,
    # 1) smoothing (in bins, not units)
    sigma_t_bins: float = 1.5,     # ~1–2 time bins mild blur
    sigma_f_bins: float = 1.0,     # ~1 freq bin mild blur
    # 2) baseline for Z (columns boolean mask over T)
    baseline_mask: np.ndarray | None = None,
    # 3) thresholding (use Z by default)
    use_z: bool = True,
    z_thresh: float = 2.5,         # ~p≈0.012 two-sided per bin
    db_thresh: float = 2.0,        # if use_z=False, threshold in dB
    # 4) cluster filtering (min size in real units)
    fs: float | None = None,       # sampling rate (Hz) to derive dt if t is seconds
    hop_s: float | None = None,    # optional: explicit time step of STFT centers (sec)
    min_duration_s: float = 0.10,  # drop blips shorter than 100 ms
    min_bandwidth_hz: float = 8.0, # drop bands narrower than 8 Hz
    # morphology
    close_iters: int = 1,          # light closing to fill pinholes (set 0 to disable)
):
    """
    Returns: smoothed_dB, Z_map, mask (boolean [F x T]) of 'significant' clusters
    """
    A_db = np.asarray(A_db, float)
    F, T = A_db.shape
    assert len(f) == F

    # ----- 1) mild smoothing (don’t overdo!)
    A_smooth = gaussian_filter(A_db, sigma=(sigma_f_bins, sigma_t_bins))

    # ----- 2) build Z-map using baseline columns (pre-stim)
    if use_z:
        if baseline_mask is None:
            # default: first 10% of time bins as baseline
            bcols = int(max(2, 0.10 * T))
            baseline_mask = np.zeros(T, bool); baseline_mask[:bcols] = True
        mu = np.nanmean(A_db[:, baseline_mask], axis=1, keepdims=True)
        sd = np.nanstd (A_db[:, baseline_mask], axis=1, keepdims=True, ddof=1)
        sd = np.where(sd <= 1e-12, 1.0, sd)
        Z = (A_db - mu) / sd
        Z_smooth = gaussian_filter(Z, sigma=(sigma_f_bins, sigma_t_bins))
        base_map = Z_smooth
        thr = z_thresh
        print(z_thresh)
    else:
        Z = None
        base_map = A_smooth
        thr = db_thresh

    # ----- 3) hard threshold
    mask = base_map >= thr

    # ----- 4) cluster-size filter (keep only “big enough” blobs)
    # derive dt (sec) and df (Hz)
    if len(t) > 1 and np.nanmax(t) > 1.1:  # looks like seconds
        dt = float(np.median(np.diff(t)))
    elif hop_s is not None:
        dt = float(hop_s)
    elif fs is not None:
        # fallback: rough hop from typical STFT (you can pass hop_s explicitly)
        dt = 0.025
    else:
        # normalized time (%): use bin counts only
        dt = None

    df = float(np.median(np.diff(f))) if len(f) > 1 else None

    # convert real-unit minima to bins
    min_tbins = int(np.ceil(min_duration_s / dt)) if (dt is not None) else 3
    min_fbins = int(np.ceil(min_bandwidth_hz / df)) if (df is not None) else 2
    min_area  = max(1, min_tbins * min_fbins)

    # 8-connected neighborhood in 2D
    conn = generate_binary_structure(2, 2)
    lab, nlab = label(mask, structure=conn)
    if nlab > 0:
        keep = np.zeros(nlab + 1, dtype=bool)
        for k in range(1, nlab + 1):
            idx = (lab == k)
            # area in bins
            area = int(idx.sum())
            # enforce minimum span both in time and frequency
            freq_span = idx.any(axis=1).sum()
            time_span = idx.any(axis=0).sum()
            if area >= min_area and freq_span >= min_fbins and time_span >= min_tbins:
                keep[k] = True
        mask = keep[lab]

    # optional light closing (dilate then erode) to fill pinholes without merging far blobs
    if close_iters > 0:
        from scipy.ndimage import binary_closing
        mask = binary_closing(mask, structure=conn, iterations=close_iters)

    return A_smooth, (Z if use_z else None), mask




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

    # use edges for crisp alignment
    t_edges = _centers_to_edges(x)
    f_edges = _centers_to_edges(f)

    plt.figure(figsize=(8, 5))

    # ---- Base spectrogram ----
    mesh = plt.pcolormesh(
        t_edges, f_edges, A,
        shading="flat", cmap="bwr",
        vmin=params.vmin, vmax=params.vmax
    )
    cb = plt.colorbar(mesh); cb.set_label("Power change (dB)")

    title = f"{patient_id} – {condition} – {reref_type}-ref ERSP: {chan_name}"
    if mode == "TN":
        pB, pS, pP = ersp["meta"]["proportions"]
        title += f" ({int(round(100*pB))}/{int(round(100*pS))}/{int(round(100*pP))}% time-normalized)"
    plt.title(title); plt.ylabel("Frequency (Hz)"); plt.ylim(0, params.fmax)

    if mode == "RT":
        plt.xlabel("Time (s) relative to onset")
        plt.axvline(0.0, color="k", ls="-", lw=1.5, label="stim")
        if mk.get("offset") is not None:
            plt.axvline(mk["offset"], color="k", ls="--", lw=1.5, label="resp")
    else:
        plt.xlabel("Normalized time (%)")
        nB, nS, nP = ersp["meta"]["bins"]
        i1 = nB; i2 = nB + nS
        seam_on  = t_edges[i1]
        seam_off = t_edges[i2]
        ticks = [t_edges[0], seam_on, seam_off, t_edges[-1]]
        plt.xticks(ticks, [f"{t:.0f}" for t in [0.0, mk['onset'], mk['offset'], 100.0]])
        plt.axvline(seam_on,  color="k", ls="-",  lw=1.5, label=f"Stim ({mk['onset']:.0f}%)")
        plt.axvline(seam_off, color="k", ls="--", lw=1.5, label=f"Resp ({mk['offset']:.0f}%)")

    # ---- NEW: mask insignificant regions ----
    if "mask" in ersp and ersp["mask"] is not None:
        mask = ersp["mask"]
        # White overlay for non-significant areas
        insignif = ~mask
        plt.pcolormesh(
            t_edges, f_edges, insignif.astype(float),
            shading="flat", cmap="Greys", vmin=0, vmax=1,
            alpha=0.6  # adjust opacity of the wash-out
        )

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
        np.savez_compressed(
            sidecar,
            avg_db=ersp["avg_db"], avg_z=ersp["avg_z"],
            f=f, x=x, markers=mk, meta=ersp["meta"],
            mask=ersp.get("mask", None)
        )
    except Exception:
        pass
    return fname


# # ----------------------------
# # High-Gamma (per-trial; RT/TN)
# # ----------------------------
# def compute_hg_trials(
#     signals: np.ndarray,
#     fs: float,
#     onsets: np.ndarray,
#     offsets: np.ndarray | None,
#     channel_idx: int,
#     *,
#     trial_ends: np.ndarray | None = None,
#     mode: str = "TN",                         # "RT" or "TN"
#     time_window: tuple[float,float] = (-1.0, 3.0),  # RT only
#     n_time_bins: int = 300,                   # for TN
#     proportions: tuple[float,float,float] = (0.0, 0.5, 0.5),
#     baseline_w: tuple[float,float] = (-0.4, 0.0),
#     baseline_calc_w: tuple[float,float] | None = (-0.4, -0.1),
#     hg_band: tuple[float,float] = (70.0, 150.0),
#     smooth_ms: float = 50.0
# ) -> dict:
#     """
#     Returns:
#       dict with:
#         'trials_z' : (n_trials, n_bins) HG z-score per trial
#         'x'        : time axis (seconds for RT, % for TN)
#         'markers'  : dict with onset/offset (seconds for RT, % for TN)
#         'meta'     : mode, fs info, etc.
#     """
#     assert signals.ndim == 2, "signals must be 2D"
#     sig = signals[:, int(channel_idx)].astype(float)
#     # HG bandpass (zero-phase)
#     nyq = 0.5 * float(fs)
#     lo, hi = max(1.0, hg_band[0]) / nyq, min(nyq - 1.0, hg_band[1]) / nyq
#     lo = max(lo, 1e-6); hi = min(hi, 0.999999)
#     b, a = butter(4, [lo, hi], btype="bandpass")
#     x_f = filtfilt(b, a, sig)
#     env = np.abs(hilbert(x_f))

#     # optional smoothing
#     if smooth_ms and smooth_ms > 0:
#         k = max(1, int(round((smooth_ms / 1000.0) * fs)))
#         if k > 1:
#             win = np.ones(k, float) / k
#             env = np.convolve(env, win, mode="same")

#     on = np.asarray(onsets, int)
#     off = np.asarray(offsets, int) if (offsets is not None and len(offsets)) else np.array([], int)
#     te  = np.asarray(trial_ends, int) if trial_ends is not None else None

#     # ---- RT (fixed window) ----
#     if str(mode).upper() == "RT":
#         # Build common grid
#         nT = max(2, int(round((time_window[1] - time_window[0]) * fs)))
#         x = np.linspace(time_window[0], time_window[1], nT, endpoint=False)
#         trials = []

#         for i in range(len(on)):
#             s = int(on[i] + time_window[0]*fs)
#             e = int(on[i] + time_window[1]*fs)
#             s = max(0, s); e = min(len(env), max(s+1, e))
#             seg = env[s:e]
#             # time vector for baseline selection
#             t_rel = (np.arange(len(seg)) / fs) + time_window[0]
#             # baseline stats
#             calc = baseline_w if baseline_calc_w is None else baseline_calc_w
#             bmask = (t_rel >= calc[0]) & (t_rel < calc[1])
#             if not np.any(bmask):
#                 bmask = np.zeros(seg.size, bool); bmask[:max(2, seg.size//10)] = True
#             mu = np.mean(seg[bmask]); sd = np.std(seg[bmask], ddof=1) + 1e-12
#             z = (seg - mu) / sd
#             # regrid to common x
#             if len(z) >= 2:
#                 t_u = (np.arange(len(z)) / fs) + time_window[0]
#                 trials.append(np.interp(x, t_u, z, left=np.nan, right=np.nan))
#             else:
#                 trials.append(np.full_like(x, np.nan, dtype=float))

#         trials_z = np.vstack(trials) if trials else np.empty((0, len(x)), float)

#         # offset marker (avg stim duration)
#         if offsets is not None and len(offsets) == len(onsets) and len(onsets) > 0:
#             offset_marker = float(np.mean((np.asarray(offsets) - np.asarray(onsets)) / float(fs)))
#         else:
#             offset_marker = (time_window[1] - time_window[0]) * 0.5

#         return dict(
#             trials_z=trials_z, x=x,
#             markers=dict(onset=0.0, offset=offset_marker),
#             meta=dict(mode="RT", fs_in=float(fs))
#         )

#     # ---- TN (piecewise with proportions) ----
#     pB, pS, pP = proportions
#     tot = pB + pS + pP
#     if not np.isclose(tot, 1.0): pB, pS, pP = (pB/tot, pS/tot, pP/tot)
#     nB = max(2, int(round(pB * n_time_bins)))
#     nS = max(2, int(round(pS * n_time_bins)))
#     nP = max(2, int(round(pP * n_time_bins)))
#     nP += (n_time_bins - (nB + nS + nP))
#     i0, i1, i2, i3 = 0, nB, nB + nS, n_time_bins
#     x = np.linspace(0.0, 100.0, n_time_bins, endpoint=False)

#     trials = []
#     calc = baseline_w if baseline_calc_w is None else baseline_calc_w

#     for i in range(len(on)):
#         seg_s = int(on[i] + baseline_w[0] * fs)
#         seg_e = int(te[i]) if te is not None else int(on[i] + baseline_w[0] * fs + 2.0*fs)
#         seg_s = max(0, seg_s); seg_e = min(len(env), max(seg_s+1, seg_e))
#         seg = env[seg_s:seg_e]
#         t_rel = (np.arange(len(seg)) / fs) + baseline_w[0]

#         # baseline stats
#         bmask = (t_rel >= calc[0]) & (t_rel < calc[1])
#         if not np.any(bmask):
#             bmask = np.zeros(seg.size, bool); bmask[:max(2, seg.size//10)] = True
#         mu = np.mean(seg[bmask]); sd = np.std(seg[bmask], ddof=1) + 1e-12
#         z = (seg - mu) / sd

#         # boundaries (seconds rel to onset)
#         off_t = ((off[i] - on[i]) / fs) if i < len(off) else np.nan
#         if not np.isfinite(off_t) or off_t <= 0:
#             val = (off[:min(len(off), len(on))] - on[:min(len(off), len(on))]) / fs
#             val = val[np.isfinite(val) & (val > 0)]
#             off_t = float(np.median(val)) if val.size else 0.5
#         next_on_t = ((te[i] - on[i]) / fs) if (te is not None) else (off_t + 2.0)

#         eps = 1.0 / fs
#         off_t = max(eps, float(off_t))
#         next_on_t = max(off_t + eps, float(next_on_t))

#         # piece masks
#         mask_base = (t_rel >= baseline_w[0]) & (t_rel < 0.0)
#         mask_stim = (t_rel >= 0.0) & (t_rel < off_t)
#         mask_post = (t_rel >= off_t) & (t_rel <= next_on_t)

#         # warp each piece to target bins
#         out = np.full((n_time_bins,), np.nan, float)
#         def _warp_1d(t_src, y_src, a0, a1, j0, j1):
#             if j1 <= j0 or t_src.size < 2: return
#             t_u, idx = np.unique(t_src, return_index=True)
#             if t_u.size < 2 or t_u[-1] <= t_u[0]: return
#             y_u = y_src[idx]
#             u = np.linspace(a0, a1, j1 - j0, endpoint=False)
#             out[j0:j1] = np.interp(u, t_u, y_u, left=np.nan, right=np.nan)

#         _warp_1d(t_rel[mask_base], z[mask_base], baseline_w[0], 0.0, i0, i1)
#         _warp_1d(t_rel[mask_stim], z[mask_stim], 0.0, off_t, i1, i2)
#         _warp_1d(t_rel[mask_post], z[mask_post], off_t, next_on_t, i2, i3)

#         trials.append(out)

#     trials_z = np.vstack(trials) if trials else np.empty((0, n_time_bins), float)

#     return dict(
#         trials_z=trials_z, x=x,
#         markers=dict(onset=100.0*(pB), offset=100.0*(pB+pS)),
#         meta=dict(mode="TN", bins=(nB,nS,nP), proportions=(pB,sP,pP))  # NOTE: typo fixed below
#     )

# # --- High-gamma trials plot (newer summary plot) ---
# def plot_hg_trials(
#     hg: dict,
#     *,
#     patient_id: str,
#     condition: str,
#     reref_type: str,
#     chan_name: str,
#     save_dir: str | None = None
# ) -> str:
#     """
#     Plots trial×time heatmap of HG z. Also overlays the across-trial mean trace.
#     """
#     Z = hg["trials_z"]      # (trials, time)
#     x = hg["x"]
#     mk = hg["markers"]
#     mode = hg["meta"]["mode"]

#     if Z.size == 0:
#         return ""

#     mean_trace = np.nanmean(Z, axis=0)

#     plt.figure(figsize=(8,6))
#     # top: mean trace
#     ax1 = plt.subplot2grid((4,1), (0,0), rowspan=1)
#     ax1.plot(x, mean_trace, lw=1.3)
#     if mode == "RT":
#         ax1.axvline(0.0, color="k", ls="-", lw=1.2)
#         ax1.axvline(mk.get("offset", 0.0), color="k", ls="--", lw=1.2)
#         ax1.set_xlabel("")
#         ax1.set_xlim([x[0], x[-1]])
#         ax1.set_ylabel("HG z (mean)")
#     else:
#         ax1.axvline(mk["onset"],  color="k", ls="-",  lw=1.2)
#         ax1.axvline(mk["offset"], color="k", ls="--", lw=1.2)
#         ax1.set_xlabel("")
#         ax1.set_xlim([x[0], x[-1]])
#         ax1.set_ylabel("HG z (mean)")

#     # bottom: trials heatmap
#     ax2 = plt.subplot2grid((4,1), (1,0), rowspan=3)
#     im = ax2.imshow(Z, aspect="auto", interpolation="nearest",
#                     extent=[x[0], x[-1], 1, Z.shape[0]], origin="lower", cmap="viridis")
#     cb = plt.colorbar(im, ax=ax2); cb.set_label("HG z")
#     if mode == "RT":
#         ax2.axvline(0.0, color="k", ls="-", lw=1.2)
#         ax2.axvline(mk.get("offset", 0.0), color="k", ls="--", lw=1.2)
#         ax2.set_xlabel("Time (s) rel. onset")
#     else:
#         ax2.axvline(mk["onset"],  color="k", ls="-",  lw=1.2)
#         ax2.axvline(mk["offset"], color="k", ls="--", lw=1.2)
#         ax2.set_xlabel("Normalized time (%)")
#     ax2.set_ylabel("Trials")
#     title = f"{patient_id} – {condition} – {reref_type}-ref HG trials: {chan_name} ({mode})"
#     plt.suptitle(title); plt.tight_layout(rect=[0,0,1,0.97])

#     fname = f"{patient_id}_{condition}_{reref_type}_HGtrials_{chan_name}{('_TN' if mode=='TN' else '')}.tif"
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         fname = os.path.join(save_dir, fname)
#     plt.savefig(fname, dpi=300, transparent=True, format="tiff")
#     plt.close()

#     # sidecar
#     sidecar = fname.replace(".tif", "_metrics.npz")
#     try:
#         np.savez_compressed(sidecar, trials_z=Z, x=x, markers=mk, meta=hg["meta"])
#     except Exception:
#         pass
#     return fname


# --- High-gamma trials plot (old style: RT-like, sorted, bwr colormap) ---
def plot_hg_trials(
    signals,
    fs,
    onsets,
    offsets,
    channel_idx,
    *,
    chan_name=None,               # if provided, used in title/filename
    channel_names=None,           # optional list; used iff chan_name is None
    patient_id="",
    condition="",
    reref_type="WM",
    time_window=(-1.0, 3.0),      # only used for last-trial fallback when no offset/trial_end
    baseline_w=(-0.6, -0.1),
    hg_band=(70.0, 150.0),
    smooth_ms=25,
    vmin=-5.0, vmax=5.0,
    figsize=(8, 5), dpi=300,
    save_dir=None,
    add_separators=False,
    sort_ascending=True,
    trial_end_indices=None,       # supports trial-end-based segmentation
    sort_by="stim"                # "stim" | "resp" | "total" | "none"
):
    """
    Mirrors the legacy HG plot (color/shape/sorting) but lives inside lf_ersp.py.
    - Sorts trials by stim/resp/total duration (or keeps chronological).
    - Draws bwr heatmap (z-scored HG envelope), onset line at t=0,
      magenta dashed offset ticks, optional separators, and trial numbers.
    - Uses trial_end when provided; otherwise falls back conservatively.
    Returns: saved TIFF filename (str).
    """
    import numpy as _np
    import matplotlib.pyplot as _plt
    from scipy.signal import butter, filtfilt, hilbert

    fs = float(fs)
    sig = signals[:, int(channel_idx)].astype(float)
    onsets   = _np.asarray(onsets,  dtype=int)
    offsets  = _np.asarray(offsets, dtype=int) if (offsets is not None and len(offsets)) else _np.array([], dtype=int)
    trialend = _np.asarray(trial_end_indices, dtype=int) if trial_end_indices is not None else None

    # --- validation / trimming
    if onsets.size == 0:
        raise ValueError("plot_hg_trials: no onsets provided.")
    if offsets.size and offsets.size != onsets.size:
        n = min(onsets.size, offsets.size)
        onsets, offsets = onsets[:n], offsets[:n]
    if trialend is not None and trialend.size != onsets.size:
        n = min(onsets.size, trialend.size)
        onsets  = onsets[:n]
        if offsets.size: offsets = offsets[:n]
        trialend = trialend[:n]

    n_trials = onsets.size

    # --- filter & smoothing (HG envelope)
    nyq = 0.5 * fs
    low = max(1.0, hg_band[0]) / nyq
    high = min(hg_band[1] / nyq, 0.999)
    if not (0 < low < high < 1):
        raise ValueError("plot_hg_trials: hg_band must be within (0, fs/2).")
    b, a = butter(4, [low, high], btype="bandpass")

    k = max(1, int(round((smooth_ms / 1000.0) * fs)))
    if k % 2 == 0:
        k += 1
    kernel = _np.ones(k, dtype=float) / k if k > 1 else None

    # --- per-trial HG z (relative to baseline_w), RT-like alignment
    trials_z, t_vecs, off_rel_s = [], [], []
    used_post_s, used_total_s = [], []

    for i, on in enumerate(onsets):
        # segment bounds
        s = max(0, on + int(round(baseline_w[0] * fs)))
        if trialend is not None:
            e = int(trialend[i])
        else:
            if i < n_trials - 1:
                e = int(onsets[i + 1])
            else:
                if i < offsets.size:
                    e = int(offsets[i] + 2.0 * fs)  # tail after last offset
                else:
                    e = int(on + (time_window[1] - time_window[0]) * fs)
        e = max(s + 1, min(e, len(sig)))
        seg = sig[s:e]

        # relative time to onset (seconds)
        t_rel = ( _np.arange(len(seg)) + s - on ) / fs

        # HG envelope
        bp  = filtfilt(b, a, seg, method="gust")
        env = _np.abs(hilbert(bp))
        if kernel is not None and env.size >= k:
            env = _np.convolve(env, kernel, mode="same")

        # baseline z
        b0, b1 = baseline_w
        bmask = (t_rel >= b0) & (t_rel < b1)
        if not _np.any(bmask):
            nT = env.shape[0]; m = max(1, nT // 10)
            bmask = _np.zeros(nT, dtype=bool); bmask[:m] = True
        mu = float(_np.mean(env[bmask])); sd = float(_np.std(env[bmask], ddof=1)) or 1.0
        z = (env - mu) / sd

        trials_z.append(z)
        t_vecs.append(t_rel)

        # markers (offset rel to onset)
        if i < offsets.size:
            off_rel_s.append( (offsets[i] - on) / fs )
            used_post_s.append( max(0.0, (e - offsets[i]) / fs) )
        else:
            off_rel_s.append(_np.nan)
        used_total_s.append( max(0.0, (e - on) / fs) )

    # --- pad to common length for plotting; build common time axis
    max_len = max(len(z) for z in trials_z)
    trial_matrix = _np.full((len(trials_z), max_len), _np.nan, dtype=float)
    for r, z in enumerate(trials_z):
        trial_matrix[r, :len(z)] = z
    t_min = min(tv[0] for tv in t_vecs)
    t_max = max(tv[-1] for tv in t_vecs)
    t_common = _np.linspace(t_min, t_max, max_len, endpoint=False)

    # --- sorting metric
    sb = (sort_by or "stim").lower()
    metric = None
    can_resp  = (trialend is not None) and (offsets.size == onsets.size) and (n_trials > 0)
    can_total = (trialend is not None) and (n_trials > 0)
    can_stim  = (offsets.size == onsets.size) and (n_trials > 0)

    if sb == "resp" and can_resp:
        metric = (trialend - offsets) / fs
    elif sb == "total" and can_total:
        metric = (trialend - onsets) / fs
    elif sb == "stim" and can_stim:
        metric = (offsets - onsets) / fs
    elif sb == "none":
        metric = None
    else:
        metric = (offsets - onsets) / fs if can_stim else None

    if metric is not None:
        fill_val = (_np.inf if sort_ascending else -_np.inf)
        metric_f = _np.nan_to_num(metric.astype(float), nan=fill_val)
        order = _np.argsort(metric_f) if sort_ascending else _np.argsort(-metric_f)
    else:
        order = _np.arange(n_trials)

    trial_matrix = trial_matrix[order, :]
    off_rel_sorted = _np.asarray(off_rel_s, dtype=float)[order]

    # --- labels & output name
    if chan_name is None:
        if (channel_names is not None) and (0 <= channel_idx < len(channel_names)):
            chan_name = str(channel_names[channel_idx])
        else:
            chan_name = f"Ch{channel_idx}"
    title = f"{patient_id} – {condition} – {reref_type}-ref HG trials: {chan_name}"
    fname = f"{patient_id}_{condition}_{reref_type}_HGtrials_{chan_name}.tif"
    out_path = os.path.join(save_dir, fname) if save_dir else fname
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # --- plot (bwr, onset line, magenta dashed offset ticks, trial numbers)
    cmap = _plt.get_cmap("bwr").copy()
    cmap.set_bad(alpha=0.0)

    _plt.figure(figsize=figsize, dpi=dpi)
    im = _plt.imshow(
        trial_matrix, aspect="auto", origin="lower",
        extent=[t_common[0], t_common[-1], 1, trial_matrix.shape[0] + 1],
        cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
    )

    # onset at 0 s
    _plt.axvline(x=0.0, color="k", linestyle="-", linewidth=1.2, zorder=3)

    # dashed offset ticks
    half_ms = 30.0
    y_rows = _np.arange(1, trial_matrix.shape[0] + 1)
    if _np.any(_np.isfinite(off_rel_sorted)):
        for y, x_ in zip(y_rows, off_rel_sorted):
            if _np.isfinite(x_):
                _plt.plot([x_ - half_ms/1000.0, x_ + half_ms/1000.0], [y, y],
                          linestyle="--", color="m", lw=2.0, solid_capstyle="butt", zorder=4)

    # annotate original row index (1-based) at baseline start
    for new_row, orig_idx in enumerate(order):
        _plt.text(baseline_w[0] + 0.05, new_row + 1,
                  str(int(orig_idx) + 1), fontsize=5, color="k",
                  va="center", ha="left")

    # optional separators
    if add_separators and trial_matrix.shape[0] > 1:
        for y in range(1, trial_matrix.shape[0]):
            _plt.hlines(y + 0.5, t_common[0], t_common[-1],
                        colors="k", linestyles=":", linewidth=0.4, alpha=0.6, zorder=2)

    cb = _plt.colorbar(im); cb.set_label("High-gamma z-score")
    _plt.title(title + (f"  | sorted by: {sb}" if sb else ""))
    _plt.xlabel("Time (s) relative to onset")
    _plt.ylabel("Trials (sorted)")
    _plt.xlim(t_common[0], t_common[-1])
    _plt.tight_layout()

    _plt.savefig(out_path, dpi=dpi, transparent=True, format="tiff")
    _plt.close()
    return out_path
