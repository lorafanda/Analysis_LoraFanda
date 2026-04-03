# lf_ersp.py — lean edition + PSD bad-channel suggestion + WM ref helper
from __future__ import annotations
from dataclasses import dataclass
import os, numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch
from scipy.ndimage import (gaussian_filter, label, generate_binary_structure,
                           binary_closing, uniform_filter)

# ----------------------------
# Parameters
# ----------------------------
@dataclass
class ERSPParams:
    nperseg: int = 128
    noverlap: int = 96                 # ~75% overlap
    nfft: int | None = None            # default -> nperseg (clamped to 2*nperseg)
    baseline_w: tuple[float,float] = (-0.4, 0.0)
    baseline_calc_w: tuple[float,float] | None = (-0.4, -0.1)
    proportions: tuple[float,float,float] = (0.0, 0.50, 0.50)  # base/stim/post (TN)
    n_time_bins: int = 400
    vmin: float = -7.0
    vmax: float =  7.0
    fmax: float = 400.0

# ----------------------------
# Utilities
# ----------------------------
def _to_khz_resampled(x: np.ndarray, fs_in: float):
    frac = Fraction((1000.0 / fs_in)).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    if up == 1 and down == 1:
        return x.astype(float), float(fs_in), 1.0
    from scipy.signal import resample_poly
    y = resample_poly(x.astype(float), up=up, down=down)
    return y, float(fs_in) * (up / down), (up / down)

def _centers_to_edges(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, float)
    if c.size == 1:
        return np.array([c[0]-0.5, c[0]+0.5], float)
    step = np.diff(c)
    return np.concatenate(([c[0]-step[0]/2], (c[:-1]+c[1:])/2, [c[-1]+step[-1]/2]))

def _spectro(seg: np.ndarray, fs: float, p: ERSPParams):
    nfft = p.nfft if p.nfft is not None else p.nperseg
    nfft = max(p.nperseg, min(nfft, 2*p.nperseg))
    f, t, Sxx = spectrogram(
        seg, fs=fs, window='hann',
        nperseg=p.nperseg, noverlap=p.noverlap, nfft=nfft,
        detrend=False, scaling="density", mode="psd"
    )
    Sxx = np.maximum(Sxx, 1e-20)
    return f, t, 10.0*np.log10(Sxx)

# ----------------------------
# PSD-based bad-channel suggestion (per patient, whole recording)
# ----------------------------
def suggest_bad_channels_by_psd(
    signals: np.ndarray, fs: float, names: list[str],
    *, mains_base: float = 50.0, fmax: float = 400.0,
    line_half_hz: float = 1.0, guard_hz: float = 3.0,
    line_z: float = 3.0, total_power_z: float = 3.5,
    flat_var_thresh: float = 1e-10
) -> tuple[list[str], dict]:
    """
    Heuristics (channel-level):
      - Very low variance (flat/broken)
      - Excessive total power (z > total_power_z)
      - Abnormally large line-harmonic power (z > line_z), using ±line_half_hz windows
        around k*mains_base up to fmax, excluding ±guard_hz around each line when
        computing the 'non-line' baseline.

    Returns:
      bad_list: list of channel names flagged
      info: dict with per-channel metrics to inspect if needed
    """
    signals = np.asarray(signals, float)
    fs = float(fs)
    n_ch = signals.shape[1]
    if n_ch != len(names):
        raise ValueError("names length must match signals.shape[1].")

    # Welch PSD (2 s windows by default, 50% overlap)
    nper = int(2.0*fs)
    nover = int(0.5*nper)
    f, Pxx = welch(signals, fs=fs, nperseg=nper, noverlap=nover, axis=0, detrend=False)
    # Restrict to [0, fmax]
    keep = f <= float(fmax)
    f, Pxx = f[keep], Pxx[keep, :]
    Pxx = np.maximum(Pxx, 1e-20)

    # Total power per channel (0..fmax)
    tot_pow = np.trapz(Pxx, f, axis=0)

    # Line-harmonic power per channel
    harmonics = [k*mains_base for k in range(1, int(fmax//mains_base)+1)]
    mask_line = np.zeros_like(f, dtype=bool)
    for h in harmonics:
        mask_line |= (f >= (h - line_half_hz)) & (f <= (h + line_half_hz))

    # baseline (non-line) region for comparison (exclude a larger guard)
    mask_guard = np.zeros_like(f, dtype=bool)
    for h in harmonics:
        mask_guard |= (f >= (h - guard_hz)) & (f <= (h + guard_hz))
    mask_non_line = ~mask_guard

    line_pow = np.trapz(Pxx[mask_line, :], f[mask_line], axis=0) if np.any(mask_line) else np.zeros(n_ch)
    nonline_pow = np.trapz(Pxx[mask_non_line, :], f[mask_non_line], axis=0) if np.any(mask_non_line) else np.ones(n_ch)
    line_ratio = line_pow / np.maximum(nonline_pow, 1e-20)

    # Simple across-channel z-scores
    def _z(x):
        mu, sd = float(np.mean(x)), float(np.std(x, ddof=1) or 1.0)
        return (x - mu) / sd
    z_tot   = _z(tot_pow)
    z_line  = _z(line_ratio)

    # Flatness (time-domain variance)
    ch_var = np.var(signals, axis=0)

    # Rules
    bad = (z_tot > total_power_z) | (z_line > line_z) | (ch_var < flat_var_thresh)
    bad_list = [names[i] for i in np.where(bad)[0]]

    info = dict(
        f=f, Pxx=Pxx, tot_pow=tot_pow, z_tot=z_tot, line_ratio=line_ratio, z_line=z_line,
        ch_var=ch_var, params=dict(mains_base=mains_base, fmax=fmax, line_half_hz=line_half_hz,
                                   guard_hz=guard_hz, line_z=line_z, total_power_z=total_power_z,
                                   flat_var_thresh=flat_var_thresh)
    )
    return bad_list, info

# ----------------------------
# Apply WM reference excluding bad contacts
# ----------------------------
def apply_wm_reference_with_exclusions(
    signals: np.ndarray, names: list[str], wm_indices: list[int],
    bad_channels_for_ref: list[str], *, method: str = "median", min_wm: int = 3
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Build a WM reference from *good* WM contacts only.
    - Excludes any WM channel whose name appears in bad_channels_for_ref.
    - method: "median" (robust) or "mean".
    Returns:
      signals_reref, used_wm_names, excluded_wm_names
    """
    if not wm_indices:
        return signals, [], []
    wm_names = [names[i] for i in wm_indices]
    keep_mask = np.array([nm not in set(bad_channels_for_ref) for nm in wm_names], bool)
    used_idx = [wm_indices[i] for i, k in enumerate(keep_mask) if k]
    excl_idx = [wm_indices[i] for i, k in enumerate(keep_mask) if not k]
    used_names = [names[i] for i in used_idx]
    excl_names = [names[i] for i in excl_idx]

    if len(used_idx) < int(min_wm):
        # Not enough clean WM contacts -> leave as is
        return signals, [], wm_names  # all effectively excluded

    X = signals[:, used_idx].astype(float)
    if method.lower() == "mean":
        ref = np.mean(X, axis=1, keepdims=True)
    else:
        ref = np.median(X, axis=1, keepdims=True)
    out = signals - ref
    return out, used_names, excl_names

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
    trial_ends: np.ndarray | None = None,      # TN needs this
    mode: str = "TN",                           # "RT" or "TN"
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

    # ---- RT
    if str(mode).upper() == "RT":
        trials_db, trials_z = [], []
        f_first = None
        for on in on_ds:
            s = int(on + time_window[0]*fs_ds)
            e = int(on + time_window[1]*fs_ds)
            s = max(0, s); e = min(len(sig_ds), max(s+1, e))
            seg = sig_ds[s:e]
            f, t_rel, S_db = _spectro(seg, fs_ds, params)
            if f_first is None: f_first = f
            t_abs = t_rel + time_window[0]
            bmask = (t_abs >= calc_w[0]) & (t_abs < calc_w[1])
            if not np.any(bmask):
                nT = S_db.shape[1]; bmask = np.zeros(nT, bool); bmask[:max(2, nT//10)] = True
            mu = np.mean(S_db[:, bmask], axis=1, keepdims=True)
            sd = np.std (S_db[:, bmask], axis=1, keepdims=True, ddof=1)
            trials_db.append((t_abs, S_db - mu))
            trials_z .append((t_abs, (S_db - mu)/(sd + 1e-12)))

        max_cols = max(A.shape[1] for _, A in trials_db) if trials_db else 0
        if max_cols == 0:
            raise ValueError("No valid trials to average.")
        x = np.linspace(time_window[0], time_window[1], max_cols, endpoint=False)

        def _regrid(pairs):
            mats = []
            for t_i, A_i in pairs:
                out = np.full((A_i.shape[0], x.size), np.nan)
                t_u, idx = np.unique(t_i, return_index=True)
                if t_u.size >= 2 and t_u[-1] > t_u[0]:
                    A_u = A_i[:, idx]
                    for r in range(A_u.shape[0]):
                        out[r, :] = np.interp(x, t_u, A_u[r, :], left=np.nan, right=np.nan)
                mats.append(out)
            with np.errstate(invalid="ignore"):
                return np.nanmean(np.stack(mats, 0), axis=0)

        avg_db = _regrid(trials_db)
        avg_z  = _regrid(trials_z)
        avg_db[:, -1] = np.nan
        avg_z[:,  -1] = np.nan

        if offsets is not None and len(offsets) == len(onsets) and len(onsets) > 0:
            offset_marker = float(np.mean((np.asarray(offsets) - np.asarray(onsets)) / float(fs)))
        else:
            offset_marker = (time_window[1] - time_window[0]) * 0.5

        print(f"[ERSP RT] matrix size: time_bins={x.size}, freq_bins={f_first.size}  ->  {x.size}×{f_first.size}")

        return dict(
            avg_db=avg_db, avg_z=avg_z, f=f_first, x=x,
            markers=dict(onset=0.0, offset=offset_marker),
            meta=dict(mode="RT", fs_in=float(fs), fs_ds=float(fs_ds), scale=float(scale))
        )

    # ---- TN
    pB, pS, pP = params.proportions
    tot = pB + pS + pP
    if not np.isclose(tot, 1.0): pB, pS, pP = (pB/tot, pS/tot, pP/tot)
    N = int(params.n_time_bins)

    def _nb(prop): return 0 if prop <= 0 else max(2, int(round(prop * N)))
    nB, nS, nP = _nb(pB), _nb(pS), _nb(pP)
    nP += (N - (nB + nS + nP))
    i0, i1, i2, i3 = 0, nB, nB + nS, N
    x = np.linspace(0.0, 100.0, N, endpoint=False)

    warped_db, warped_z = [], []
    f_common = None

    for i in range(len(on_ds)):
        on = on_ds[i]
        s = on + int(round(base_w[0]*fs_ds))
        e = int(te_ds[i])
        s = max(0, s); e = min(len(sig_ds), max(s+1, e))
        seg = sig_ds[s:e]

        f, t_rel, S_db = _spectro(seg, fs_ds, params)
        t_rel = t_rel + base_w[0]
        if f_common is None: f_common = f

        calc = params.baseline_w if params.baseline_calc_w is None else params.baseline_calc_w
        bmask = (t_rel >= calc[0]) & (t_rel < calc[1])
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

        Wdb = np.full((Srel.shape[0], N), np.nan); Wz = np.full((Srel.shape[0], N), np.nan)

        def _warp_piece(t_src, A_src, j0, j1, a0, a1, W):
            if j1 <= j0 or t_src.size < 2: return
            t_u, idx = np.unique(t_src, return_index=True)
            if t_u.size < 2 or t_u[-1] <= t_u[0]: return
            A_u = A_src[:, idx]
            u = np.linspace(a0, a1, j1 - j0, endpoint=False)
            for r in range(A_u.shape[0]):
                W[r, j0:j1] = np.interp(u, t_u, A_u[r, :], left=np.nan, right=np.nan)

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

    print(f"[ERSP TN] matrix size: time_bins={x.size}, freq_bins={f_common.size}  ->  {x.size}×{f_common.size}")

    return dict(
        avg_db=avg_db, avg_z=avg_z, f=f_common, x=x,
        markers=dict(onset=100.0*(pB), offset=100.0*(pB+pS)),
        meta=dict(mode="TN", fs_in=float(fs), fs_ds=float(fs_ds), scale=float(scale),
                  proportions=(pB,pS,pP), bins=(nB,nS,nP))
    )

# ----------------------------
# Mask builder (Z two-sided, raw/processed)
# ----------------------------
def build_masks(
    A_db: np.ndarray, f: np.ndarray, t: np.ndarray,
    *, use_z: bool = True, z_thresh: float = 2.5, db_thresh: float = 2.0,
    thresh_mode: str = "two-sided",
    # smoothing controls
    sigma_t_bins: float = 1.5, sigma_f_bins: float = 1.0,
    smooth_kind: str = "gaussian",
    box_kernel: tuple[int,int] = (0,0),
    smooth_passes: int = 1,
    # bin-based cluster size
    min_time_bins: int = 1,
    min_freq_bins: int = 1,
    raw_mode: bool = False, close_iters: int = 1
):
    A_db = np.asarray(A_db, float); F, T = A_db.shape
    assert len(f) == F

    def _smooth_map(M):
        if str(smooth_kind).lower() == "box":
            kf, kt = map(int, (box_kernel or (0,0)))
            if kf > 1 or kt > 1:
                return _box_smooth2d(M, k_f=kf, k_t=kt, passes=int(max(1, smooth_passes)))
            return M
        do_gauss = (sigma_t_bins > 0) or (sigma_f_bins > 0)
        return gaussian_filter(M, sigma=(sigma_f_bins, sigma_t_bins)) if do_gauss else M

    if use_z:
        # Expect the caller to pass a true baseline Z (e.g., avg_z). If they pass dB, we fallback to a rough Z:
        bcols = int(max(2, 0.10 * T))
        bmask = np.zeros(T, bool); bmask[:bcols] = True
        mu = np.nanmean(A_db[:, bmask], axis=1, keepdims=True)
        sd = np.nanstd (A_db[:, bmask], axis=1, keepdims=True, ddof=1)
        sd = np.where(sd <= 1e-12, 1.0, sd)
        Z = (A_db - mu) / sd
        Mbase = _smooth_map(Z)
        thr = float(z_thresh)
    else:
        Z = None
        A_s = _smooth_map(A_db)
        Mbase = A_s
        thr = float(db_thresh)

    mode = str(thresh_mode).lower()
    if mode == "two-sided":
        Mpos = (Mbase >=  thr); Mneg = (Mbase <= -thr); Mbi = Mpos | Mneg
    elif mode == "neg":
        Mpos = np.zeros_like(Mbase, bool); Mneg = (Mbase <= -thr); Mbi = Mneg.copy()
    else:
        Mpos = (Mbase >=  thr); Mneg = np.zeros_like(Mbase, bool); Mbi = Mpos.copy()

    if raw_mode:
        return (A_s if not use_z else Z), (Z if use_z else None), Mbi, Mpos, Mneg

    # size filters in BINS
    min_tb = int(max(1, min_time_bins))
    min_fb = int(max(1, min_freq_bins))
    min_area = max(1, min_tb * min_fb)

    conn = generate_binary_structure(2, 2)
    def _filter(m):
        lab, nlab = label(m, structure=conn)
        if nlab == 0: return m
        keep = np.zeros(nlab + 1, bool)
        for k in range(1, nlab + 1):
            idx = (lab == k)
            time_span = idx.any(axis=0).sum()
            freq_span = idx.any(axis=1).sum()
            area      = idx.sum()
            if (time_span >= min_tb) and (freq_span >= min_fb) and (area >= min_area):
                keep[k] = True
        out = keep[lab]
        return binary_closing(out, structure=conn, iterations=int(close_iters)) if close_iters>0 else out

    return (A_s if not use_z else Mbase), (Z if use_z else None), _filter(Mbi), _filter(Mpos), _filter(Mneg)

# ----------------------------
# Plot & save
# ----------------------------
def plot_ersp(
    ersp: dict, *,
    patient_id: str, condition: str, reref_type: str, chan_name: str,
    save_dir: str | None = None, params: ERSPParams = ERSPParams()
) -> str:
    f = ersp["f"]; x = ersp["x"]; A = ersp["avg_db"]
    mode = ersp["meta"]["mode"]; mk = ersp["markers"]
    t_edges = _centers_to_edges(x); f_edges = _centers_to_edges(f)

    plt.figure(figsize=(7, 4))
    mesh = plt.pcolormesh(t_edges, f_edges, A, shading="flat", cmap="bwr",
                          vmin=params.vmin, vmax=params.vmax)
    cb = plt.colorbar(mesh); cb.set_label("Power change (dB)")

    title = f"{patient_id} – {condition} – {reref_type}-ref ERSP: {chan_name}"
    plt.ylabel("Frequency (Hz)"); plt.ylim(0, params.fmax)

    if mode == "RT":
        plt.xlabel("Time (s) rel. onset")
        plt.axvline(0.0, color="k", ls="-", lw=1.5, label="stim")
        if mk.get("offset") is not None:
            plt.axvline(mk["offset"], color="k", ls="--", lw=1.5, label="resp")
    else:
        plt.xlabel("Normalized time (%)")
        pB, pS, pP = ersp["meta"]["proportions"]
        nB, nS, _ = ersp["meta"]["bins"]
        seam_on, seam_off = t_edges[nB], t_edges[nB + nS]
        plt.xticks([t_edges[0], seam_on, seam_off, t_edges[-1]],
                   [f"{t:.0f}" for t in [0.0, mk['onset'], mk['offset'], 100.0]])
        plt.axvline(seam_on,  color="k", ls="-",  lw=1.5, label=f"Stim ({mk['onset']:.0f}%)")
        plt.axvline(seam_off, color="k", ls="--", lw=1.5, label=f"Resp ({mk['offset']:.0f}%)")
        title += f" ({int(round(100*pB))}/{int(round(100*pS))}/{int(round(100*pP))}% TN)"

    if ersp.get("mask", None) is not None:
        insignif = ~ersp["mask"]
        plt.pcolormesh(t_edges, f_edges, insignif.astype(float),
                       shading="flat", cmap="Greys", vmin=0, vmax=1, alpha=0.6)

    plt.title(title); plt.legend(loc="upper right"); plt.tight_layout()

    fname = f"{patient_id}_{condition}_{reref_type}_ERSP_{chan_name}{('_TN' if mode=='TN' else '')}.tif"
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, fname)
    plt.savefig(fname, dpi=300, transparent=True, format="tiff")
    plt.close()

    sidecar = fname.replace(".tif", "_metrics.npz")
    try:
        np.savez_compressed(
            sidecar,
            avg_db=ersp["avg_db"], avg_z=ersp["avg_z"],
            f=f, x=x, markers=mk, meta=ersp["meta"], mask=ersp.get("mask", None)
        )
    except Exception:
        pass
    return fname

def save_cluster_plots(
    ersp: dict, *,
    patient_id: str, condition: str, reref_type: str, chan_name: str,
    save_root: str,
    z_thresh: float = 2.5, use_z: bool = True, thresh_mode: str = "two-sided",
    raw: bool = False, params: ERSPParams = ERSPParams(),
    smooth_kind: str = "box", box_kernel: tuple[int,int] = (6,6), smooth_passes: int = 1
) -> tuple[str, str | None]:
    f, x, A = ersp["f"], ersp["x"], ersp["avg_db"]
    A_s, Z, M_bi, M_pos, M_neg = build_masks(
        A, f, x,
        use_z=use_z, z_thresh=z_thresh, thresh_mode=thresh_mode,
        smooth_kind=smooth_kind, box_kernel=box_kernel, smooth_passes=smooth_passes,
        sigma_t_bins=1.5, sigma_f_bins=1.0,  # ignored when box
        min_time_bins=1, min_freq_bins=1,
        raw_mode=bool(raw)
    )
    ersp_plot = dict(ersp); ersp_plot["mask"] = M_bi
    outdir = os.path.join(save_root, ("cluster raw plots" if raw else "ERSP_cluster_plots"))
    fig_path = plot_ersp(ersp_plot, patient_id=patient_id, condition=condition,
                         reref_type=reref_type, chan_name=chan_name, save_dir=outdir, params=params)
    sidecar = fig_path.replace(".tif", ("_RAWmask.npz" if raw else "_PROCmask.npz"))
    try:
        os.makedirs(outdir, exist_ok=True)
        np.savez_compressed(
            sidecar,
            mask_bipolar=M_bi, mask_pos=M_pos, mask_neg=M_neg,
            use_z=use_z, z_thresh=z_thresh, thresh_mode=thresh_mode,
            f=f, x=x, mode=ersp["meta"]["mode"], map_smoothed=(Z if use_z else A_s)
        )
    except Exception:
        sidecar = None
    return fig_path, sidecar

# ----------------------------
# Simple 2D box smoother (preserves shape)
# ----------------------------
def _box_smooth2d(A: np.ndarray, k_f: int, k_t: int, passes: int = 1) -> np.ndarray:
    A = np.asarray(A, float)
    k_f = int(max(1, k_f)); k_t = int(max(1, k_t))
    if (k_f <= 1 and k_t <= 1) or passes <= 0:
        return A
    out = A
    for _ in range(int(passes)):
        out = uniform_filter(out, size=(k_f, k_t), mode="nearest")
    return out
