"""
lf_features.py — Compact feature representations for clustering.

Currently provides the **band-aware ERSP downsampler**: takes a full
(n_freq_orig, n_time_orig) ERSP (typically 129 × 300, covering 0..~500 Hz)
and compresses it to a much smaller (n_freq_out, n_time_out) grid where:

  - the frequency axis is collapsed to a user-supplied list of neuroscience-
    motivated bands (delta, theta, alpha, beta, gamma, HG, HFO ...) — each
    output row = mean of input bins whose frequency falls in that band's
    [lo, hi) window
  - the time axis is downsampled via skimage's anti-aliased resize, same
    convention `lf_minus101.downsample_minus101_map` already uses

Why this exists:
  Raw 38,700-feature clustering is curse-of-dimensionality territory;
  silhouettes hover ~0.05 because the distance metric becomes meaningless.
  Compressing to ~15 freq × 30 time = 450 features lets KMeans and HC
  find real structure (Becht et al. 2019). The band-aware vs uniform
  spacing matters because high-gamma is much narrower than the HFO band
  but each is a distinct neurophysiological signal — we want one bin
  representing each.

Public API:
  downsample_ersp_to_bands(ersp, freq_band_edges, fmax_hz, time_bins_out)
      -> (n_freq_out, time_bins_out) ndarray

  build_X_3d_downsampled(ersp_list, ...)
      -> (n_samples, n_freq_out, time_bins_out) stack

  FREQ_BANDS_15_TO_400HZ : recommended 15-band split covering 1..400 Hz.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    from skimage.transform import resize as _sk_resize
except ImportError:  # pragma: no cover
    _sk_resize = None


# ============================================================
# Recommended band edges
# ============================================================
# 15 contiguous bands, finer at low freq, coarser at HFO. Each
# standard EEG band (delta, theta, alpha, beta, low/mid gamma, HG, HFO)
# gets at least one bin. Spans 1..400 Hz; signal above 400 Hz is
# dropped at downsample time per Lora's request.
FREQ_BANDS_15_TO_400HZ: List[Tuple[float, float]] = [
    (  1.0,   4.0),    # delta
    (  4.0,   8.0),    # theta
    (  8.0,  13.0),    # alpha
    ( 13.0,  20.0),    # low beta
    ( 20.0,  30.0),    # high beta
    ( 30.0,  50.0),    # low gamma
    ( 50.0,  70.0),    # mid gamma
    ( 70.0, 100.0),    # HG low
    (100.0, 130.0),    # HG mid
    (130.0, 170.0),    # HG high
    (170.0, 220.0),    # HFO low
    (220.0, 270.0),    # HFO mid
    (270.0, 320.0),    # HFO high
    (320.0, 360.0),    # HFO upper
    (360.0, 400.0),    # top
]


# ============================================================
# Single-sample downsample
# ============================================================
def downsample_ersp_to_bands(
    ersp: np.ndarray,
    freq_band_edges: Sequence[Tuple[float, float]] = FREQ_BANDS_15_TO_400HZ,
    *,
    fmax_hz: float = 500.0,
    time_bins_out: int = 30,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compress one ERSP to a (n_freq_out, time_bins_out) grid.

    Parameters
    ----------
    ersp : (n_freq_orig, n_time_orig) ndarray
        The full-resolution ERSP. Assumes the freq axis is uniform from
        0 to `fmax_hz` (standard STFT output).
    freq_band_edges : list of (lo_hz, hi_hz) tuples
        Output freq bins. Each output row = mean of input bins whose
        center frequency falls in [lo, hi). Should be contiguous and
        non-overlapping for cleanest interpretation, but isn't enforced.
    fmax_hz : float
        Upper edge of the original freq axis (Hz). Default 500.0 — adjust
        to your STFT's actual top frequency.
    time_bins_out : int
        Number of time bins in the output (default 30).
    verbose : bool
        If True, warn when a freq band contains zero input bins.

    Returns
    -------
    ds : (n_freq_out, time_bins_out) float32 ndarray
    """
    if _sk_resize is None:
        raise ImportError(
            "scikit-image is required for downsample_ersp_to_bands. "
            "Install with `pip install scikit-image`."
        )
    ersp = np.asarray(ersp, dtype=np.float32)
    if ersp.ndim != 2:
        raise ValueError(f"ersp must be 2D (n_freq, n_time); got shape {ersp.shape}")
    n_freq_orig, n_time_orig = ersp.shape

    # Reconstruct the original freq axis assuming uniform 0..fmax linspace
    freqs = np.linspace(0.0, float(fmax_hz), n_freq_orig)

    # Per-band mean across the freq axis
    n_freq_out = len(freq_band_edges)
    band_means = np.zeros((n_freq_out, n_time_orig), dtype=np.float32)
    for i, (lo, hi) in enumerate(freq_band_edges):
        mask = (freqs >= float(lo)) & (freqs < float(hi))
        if mask.any():
            band_means[i] = ersp[mask].mean(axis=0)
        else:
            if verbose:
                print(f"[downsample_ersp_to_bands] band ({lo}, {hi}) Hz "
                      f"has no input bins (fmax_hz={fmax_hz}, n_freq={n_freq_orig}); "
                      f"row {i} left at zero")

    # Anti-aliased resize on the time axis only
    if time_bins_out == n_time_orig:
        return band_means
    out = _sk_resize(
        band_means,
        (n_freq_out, int(time_bins_out)),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    return out


# ============================================================
# Stacked / batch version
# ============================================================
def build_X_3d_downsampled(
    ersp_list: Iterable[np.ndarray],
    *,
    freq_band_edges: Sequence[Tuple[float, float]] = FREQ_BANDS_15_TO_400HZ,
    fmax_hz: float = 500.0,
    time_bins_out: int = 30,
    verbose: bool = True,
) -> np.ndarray:
    """
    Apply `downsample_ersp_to_bands` to every ERSP in the list and stack.

    Returns
    -------
    X_3d_ds : (n_samples, n_freq_out, time_bins_out) float32 ndarray
              ready to flatten + standardize + cluster.
    """
    ersp_list = list(ersp_list)
    n = len(ersp_list)
    if n == 0:
        return np.zeros((0, len(freq_band_edges), int(time_bins_out)), dtype=np.float32)

    # Compute first one to learn output shape, then allocate + fill
    first = downsample_ersp_to_bands(
        ersp_list[0], freq_band_edges,
        fmax_hz=fmax_hz, time_bins_out=time_bins_out, verbose=verbose,
    )
    nf_out, nt_out = first.shape
    X = np.zeros((n, nf_out, nt_out), dtype=np.float32)
    X[0] = first
    for i in range(1, n):
        X[i] = downsample_ersp_to_bands(
            ersp_list[i], freq_band_edges,
            fmax_hz=fmax_hz, time_bins_out=time_bins_out, verbose=False,
        )
    if verbose:
        print(f"[build_X_3d_downsampled] {n} samples · "
              f"{nf_out} freq bins · {nt_out} time bins · "
              f"flat dim = {nf_out * nt_out}")
    return X
