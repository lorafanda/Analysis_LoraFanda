# Core PD detector (monopolar signals only, for bipolar, substract one signal from the other first)
# =========================

def pd_detect_deriv_core(
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