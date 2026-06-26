"""
lf_notch.py
===========

Adaptive mains-harmonic notch filtering for multi-channel electrophysiology,
with a genuinely **per-channel** peak model.

Why this module exists
----------------------
The original ``notch_mains_harmonics`` (in the micro/macro pipeline) decides
*which* harmonics to notch — and *how wide* each notch should be — from a single
**channel-median PSD** computed across every channel handed to it. The same set
of harmonics and the same per-harmonic ``Q`` is then applied to every channel.

That is a *per-group* model. A mains harmonic that is strong on one micro contact
but weak on the others is washed out of the median and never notched on the
contact that actually carries it; conversely a peak present on only a few contacts
can drive a notch applied to channels that didn't need it. This is the dominant
source of residual line noise leaking into individual micro channels.

This module keeps the good ideas of the original (only notch a harmonic where a
*real* spectral peak exists; set ``Q`` from the measured peak width; zero-phase
``filtfilt``) but makes the peak detection and ``Q`` selection **per channel**.
The legacy behaviour is preserved behind ``mode="median"`` so you can A/B compare
with a single flag.

Public API
----------
notch_mains_harmonics(X, fs, *, mode="per_channel", ..., return_audit=False)
    Drop-in replacement for the old function (same name, superset of knobs).
apply_notch_with_audit(signals, fs, patient_id, *, ...)
    Per-patient gated wrapper (compatible with the old call site).
harmonic_residual_db(X, fs, *, base=50.0, ...)
    Per-channel QC: residual peak-vs-floor in dB at each harmonic (want ~0 post).

Design notes
------------
* ``Q`` is derived from the half-power (-3 dB) full width at half maximum (FWHM)
  of each peak, measured on that channel's own PSD: ``Q = f0 / (df_safety * FWHM)``,
  clamped to ``[Q_min, Q_max]``. Sharper peaks -> higher Q -> narrower notch ->
  less collateral distortion of nearby signal.
* The Welch PSD uses ``average="median"`` when available, which is robust to
  transient broadband artifacts (movement, saturation) that would bias a
  mean-averaged PSD and cause spurious "peaks".
* All filtering is zero-phase (``filtfilt`` with ``method="gust"``), so spike
  waveform timing/shape is preserved.
* The module only imports numpy at top level; scipy is imported lazily inside the
  functions, so merely importing ``lf_notch`` can never fail on a scipy hiccup.
* No project-internal imports -> this module is fully standalone and safe to
  import as a top-level module (``import lf_notch``) or as a package submodule.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "notch_mains_harmonics",
    "apply_notch_with_audit",
    "harmonic_residual_db",
    "NotchDesign",
]


# -----------------------------------------------------------------------------
# A small record describing one designed notch (handy for auditing/plotting)
# -----------------------------------------------------------------------------
class NotchDesign(tuple):
    """(f0, Q, z, fwhm) for a single notched harmonic, with named access."""
    __slots__ = ()

    def __new__(cls, f0, Q, z, fwhm):
        return super().__new__(cls, (float(f0), float(Q), float(z), float(fwhm)))

    f0   = property(lambda self: self[0])
    Q    = property(lambda self: self[1])
    z    = property(lambda self: self[2])
    fwhm = property(lambda self: self[3])

    def __repr__(self):
        return f"NotchDesign(f0={self[0]:.1f}Hz, Q={self[1]:.1f}, z={self[2]:.1f}, FWHM={self[3]:.2f}Hz)"


# -----------------------------------------------------------------------------
# Core: design the notches for a single PSD
# -----------------------------------------------------------------------------
def _design_notches_for_psd(f_psd, P_db, harms, nyq, *,
                            peak_z_thresh, local_hz, peak_hz,
                            min_df, max_df, df_safety, Q_min, Q_max,
                            min_prominence_db=6.0, min_bg_bins=5):
    """
    Given one PSD (in dB), return a list of NotchDesign for the harmonics that
    show a statistically real peak.

    A harmonic ``f0`` is notched iff the peak inside ``[f0-peak_hz, f0+peak_hz]``:
      (1) rises ``>= peak_z_thresh`` std above the local background
          ``[f0-local_hz, f0+local_hz] \\ peak band`` (relative test), AND
      (2) stands ``>= min_prominence_db`` dB above that background (absolute test).

    The absolute prominence (2) is what stops over-notching at high frequencies:
    where the spectrum flattens, the background std collapses so the z-score (1)
    trips on tiny ripples; requiring a real dB prominence rejects those. The notch
    width (Q) is derived from the -3 dB FWHM of the peak.
    """
    if not np.all(np.isfinite(P_db)):
        # Flat / dead / NaN channel: nothing to notch.
        return []

    df_bin = float(f_psd[1] - f_psd[0])
    designs = []

    for f0 in harms:
        if f0 >= 0.95 * nyq:
            continue

        local  = (f_psd >= f0 - local_hz) & (f_psd <= f0 + local_hz)
        peak_w = (f_psd >= f0 - peak_hz)  & (f_psd <= f0 + peak_hz)
        bg     = local & ~peak_w

        if int(bg.sum()) < min_bg_bins or not np.any(peak_w):
            continue

        bg_vals  = P_db[bg]
        bg_mean  = float(np.mean(bg_vals))
        bg_std   = float(np.std(bg_vals) + 1e-6)

        local_idx = np.flatnonzero(local)
        pk_idx    = int(local_idx[int(np.argmax(P_db[local]))])
        pk_val    = float(P_db[pk_idx])

        prominence_db = pk_val - bg_mean
        z = prominence_db / bg_std
        if z < peak_z_thresh or prominence_db < min_prominence_db:
            continue

        # -3 dB half-power FWHM: walk out from the peak until we drop below half.
        half = pk_val - 3.0
        i = pk_idx
        while i > local_idx[0] and P_db[i] > half:
            i -= 1
        j = pk_idx
        while j < local_idx[-1] and P_db[j] > half:
            j += 1
        fwhm = max(float(f_psd[j] - f_psd[i]), 2.0 * df_bin)
        fwhm = float(np.clip(fwhm, min_df, max_df))

        Q = float(np.clip(f0 / (df_safety * fwhm), Q_min, Q_max))
        designs.append(NotchDesign(f0, Q, z, fwhm))

    return designs


def _apply_designs(y, designs, fs, repeats, iirnotch, filtfilt):
    """Apply a list of NotchDesign to a 1-D signal ``y`` (zero-phase)."""
    for d in designs:
        b, a = iirnotch(w0=d.f0, Q=d.Q, fs=fs)
        for _ in range(int(repeats)):
            y = filtfilt(b, a, y, method="gust")
    return y


# -----------------------------------------------------------------------------
# Public: the notch
# -----------------------------------------------------------------------------
def notch_mains_harmonics(X, fs, *, base=50.0, max_hz=None, repeats=1,
                          peak_z_thresh=3.0, min_prominence_db=6.0,
                          Q_min=10.0, Q_max=500.0,
                          local_hz=5.0, peak_hz=1.0, min_df=0.3, max_df=1.5,
                          df_safety=1.0, mode="per_channel", nper_s=4.0,
                          welch_average="median", min_bg_bins=5,
                          return_audit=False, verbose=True):
    """
    Adaptive mains-harmonic notch.

    Parameters
    ----------
    X : ndarray (n_samples,) or (n_samples, n_channels)
        Input signal(s). Filtered along axis 0.
    fs : float
        Sampling rate (Hz).
    base : float
        Mains fundamental (50.0 in EU, 60.0 in US).
    max_hz : float or None
        Highest harmonic to consider (default 0.98 * Nyquist).
    repeats : int
        Forward-backward ``filtfilt`` passes per harmonic (sharper attenuation).
    peak_z_thresh : float
        z-score over the local PSD background required to call a harmonic "real".
    min_prominence_db : float
        Absolute prominence (dB above local background) ALSO required to notch a
        harmonic. Prevents over-notching at high frequencies, where the spectrum
        flattens, the background std collapses, and the z-score alone trips on
        tiny ripples. 0 disables (z-only, the old behaviour). ~6 dB is a good default.
    Q_min, Q_max : float
        Clamp on the auto-derived notch Q.
    local_hz : float
        Half-width (Hz) of the local PSD window used for the background estimate.
    peak_hz : float
        Half-width (Hz) of the peak window used for the z-test.
    min_df, max_df : float
        Clamp on the measured peak FWHM (Hz).
    df_safety : float
        Multiplicative widening of the measured FWHM (1.0 = exact, 1.2 = 20% safety).
    mode : {"per_channel", "median"}
        ``"per_channel"`` (default): detect peaks and derive Q on **each channel's
        own PSD**, notch each channel with only its own significant harmonics.
        ``"median"``: legacy behaviour — design once from the channel-median PSD
        and apply the same notches to every channel (for A/B comparison).
    nper_s : float
        Welch segment length in seconds (frequency resolution = 1/nper_s Hz).
    welch_average : {"median", "mean"}
        Welch segment averaging. "median" is robust to transient artifacts;
        falls back to "mean" automatically on older scipy.
    min_bg_bins : int
        Minimum number of background frequency bins required to run the z-test.
    return_audit : bool
        If True, also return an audit dict.
    verbose : bool
        Print a per-channel (or median) summary of what was notched.

    Returns
    -------
    Y : ndarray
        Filtered signal, same shape/orientation as ``X``.
    audit : dict   (only if ``return_audit=True``)
        ``mode="per_channel"`` -> ``{channel_index: [NotchDesign, ...]}``
        ``mode="median"``      -> ``{"__median__": [NotchDesign, ...]}``
    """
    from scipy.signal import welch, iirnotch, filtfilt

    X = np.asarray(X, dtype=float)
    was_1d = (X.ndim == 1)
    if was_1d:
        X = X[:, None]
    n_samples, n_ch = X.shape

    nyq = 0.5 * fs
    lim = min(0.98 * nyq, max_hz if max_hz is not None else 0.98 * nyq)
    harms = [h * base for h in range(1, int(lim // base) + 1)]

    nper = int(min(nper_s * fs, n_samples))
    if nper < 16:
        nper = n_samples

    def _welch(x):
        try:
            return welch(x, fs=fs, nperseg=nper, detrend=False, average=welch_average)
        except TypeError:  # older scipy without `average`
            return welch(x, fs=fs, nperseg=nper, detrend=False)

    design_kw = dict(peak_z_thresh=peak_z_thresh, min_prominence_db=min_prominence_db,
                     local_hz=local_hz, peak_hz=peak_hz,
                     min_df=min_df, max_df=max_df, df_safety=df_safety,
                     Q_min=Q_min, Q_max=Q_max, min_bg_bins=min_bg_bins)

    Y = X.copy()
    audit = {}

    if mode == "median":
        f_psd, P0 = _welch(X[:, 0])
        if n_ch > 1:
            P_med = np.median(np.vstack([_welch(X[:, ci])[1] for ci in range(n_ch)]), axis=0)
        else:
            P_med = P0
        P_db = 10.0 * np.log10(P_med + 1e-30)
        designs = _design_notches_for_psd(f_psd, P_db, harms, nyq, **design_kw)
        audit["__median__"] = designs
        if designs:
            for ci in range(n_ch):
                Y[:, ci] = _apply_designs(Y[:, ci], designs, fs, repeats, iirnotch, filtfilt)
        if verbose:
            _print_median_summary(designs, n_ch)

    elif mode == "per_channel":
        for ci in range(n_ch):
            f_psd, P = _welch(X[:, ci])
            P_db = 10.0 * np.log10(P + 1e-30)
            designs = _design_notches_for_psd(f_psd, P_db, harms, nyq, **design_kw)
            audit[ci] = designs
            if designs:
                Y[:, ci] = _apply_designs(Y[:, ci], designs, fs, repeats, iirnotch, filtfilt)
        if verbose:
            _print_per_channel_summary(audit, n_ch)

    else:
        raise ValueError(f"mode must be 'per_channel' or 'median', got {mode!r}")

    result = Y.squeeze() if was_1d else Y
    return (result, audit) if return_audit else result


def _print_median_summary(designs, n_ch):
    if not designs:
        print(f"[notch:median] {n_ch} ch — no significant harmonics found")
        return
    fs_ = ", ".join(f"{d.f0:.0f}Hz(Q{d.Q:.0f})" for d in designs)
    print(f"[notch:median] {n_ch} ch — notched {len(designs)} harmonics applied to all: {fs_}")


def _print_per_channel_summary(audit, n_ch):
    counts = [len(audit.get(ci, [])) for ci in range(n_ch)]
    total = int(np.sum(counts))
    print(f"[notch:per_channel] {n_ch} channels — {total} notch operations "
          f"({np.min(counts) if counts else 0}-{np.max(counts) if counts else 0} per channel, "
          f"median {int(np.median(counts)) if counts else 0})")
    # Show which harmonics are channel-specific (present on some but not all channels)
    all_f = {}
    for ci in range(n_ch):
        for d in audit.get(ci, []):
            all_f.setdefault(round(d.f0), 0)
            all_f[round(d.f0)] += 1
    specific = [f for f, c in sorted(all_f.items()) if 0 < c < n_ch]
    if specific:
        preview = ", ".join(f"{f}Hz({all_f[f]}/{n_ch})" for f in specific[:12])
        print(f"                 channel-specific harmonics (missed by median mode): {preview}"
              + (" ..." if len(specific) > 12 else ""))


# -----------------------------------------------------------------------------
# Per-patient gated wrapper (compatible with the legacy apply_notch_with_audit)
# -----------------------------------------------------------------------------
def apply_notch_with_audit(signals, fs, patient_id, pid_raw=None, *,
                           notch_patients=(), mains_base=50.0, fmax=500.0,
                           repeats=1, peak_z_thresh=3.0, mode="per_channel",
                           verbose=True, **kw):
    """
    Adaptive mains-harmonic notch with per-patient gating.

    If ``notch_patients`` is empty, the notch is always applied. Otherwise it is
    applied only when one of the ``notch_patients`` substrings matches
    ``patient_id`` or ``pid_raw``. Extra keyword args are forwarded to
    :func:`notch_mains_harmonics`.
    """
    if notch_patients:
        hay = f"{patient_id}|{pid_raw}"
        if not any(str(s) in hay for s in notch_patients):
            if verbose:
                print(f"[notch] {patient_id}: not in notch_patients — skipped")
            return signals
    if verbose:
        print(f"[notch] {patient_id}  mode={mode}  z>={peak_z_thresh}")
    return notch_mains_harmonics(
        signals, fs, base=mains_base, max_hz=min(fmax, 0.5 * fs),
        repeats=repeats, peak_z_thresh=peak_z_thresh, mode=mode,
        verbose=verbose, **kw,
    )


# -----------------------------------------------------------------------------
# QC: per-channel residual at harmonics (use before vs after)
# -----------------------------------------------------------------------------
def harmonic_residual_db(X, fs, *, base=50.0, max_hz=800.0, nper_s=4.0,
                         peak_hz=1.0, floor_lo=2.0, floor_hi=10.0,
                         welch_average="median"):
    """
    Per-channel residual harmonic peak-vs-floor, in dB, at each harmonic.

    For each channel and each harmonic ``f0``, returns
    ``10*log10( mean(PSD in [f0+-peak_hz]) / mean(PSD in [f0+-(floor_lo..floor_hi)]) )``.
    Run on the signal *before* and *after* notching; post values should be ~0 dB
    on the channels (and harmonics) that were notched.

    Returns
    -------
    f0s : ndarray (n_harmonics,)
        Harmonic frequencies.
    resid_db : ndarray (n_channels, n_harmonics)
        Residual dB per channel per harmonic. NaN where the bands are empty.
    """
    from scipy.signal import welch

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n_samples, n_ch = X.shape
    nyq = 0.5 * fs
    lim = min(max_hz, 0.98 * nyq)
    f0s = np.array([h * base for h in range(1, int(lim // base) + 1)], dtype=float)
    nper = int(min(nper_s * fs, n_samples))
    if nper < 16:
        nper = n_samples

    def _welch(x):
        try:
            return welch(x, fs=fs, nperseg=nper, detrend=False, average=welch_average)
        except TypeError:
            return welch(x, fs=fs, nperseg=nper, detrend=False)

    resid = np.full((n_ch, f0s.size), np.nan, dtype=float)
    for ci in range(n_ch):
        f_psd, P = _welch(X[:, ci])
        for k, f0 in enumerate(f0s):
            pk = (f_psd >= f0 - peak_hz) & (f_psd <= f0 + peak_hz)
            fl = (((f_psd >= f0 - floor_hi) & (f_psd <= f0 - floor_lo)) |
                  ((f_psd >= f0 + floor_lo) & (f_psd <= f0 + floor_hi)))
            if not (np.any(pk) and np.any(fl)):
                continue
            resid[ci, k] = 10.0 * np.log10(P[pk].mean() / P[fl].mean())
    return f0s, resid
