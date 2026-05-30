"""
lf_hg.py — High-Gamma (HG) feature for clustering.

The ERSPs produced by 140 (`compute_ersp` → saved as .npy under
`outputs/04_ersp_LM_RAWONLY/<pid>/LM/ERSP_matrix/<cond>/`) are 2D
(n_freq × n_time). For HG-feature clustering, we collapse the freq axis
to a single high-gamma band time series:

    hg(t) = mean(ERSP[freq ∈ HG_BAND, t])

producing a 1D vector of length n_time (default 300) per sample.

Default HG band: 70–150 Hz (matches `cfg.hg_band` in 140's config).
Frequency axis is reconstructed assuming `linspace(0, fmax, n_freq)`
with fmax = 500 Hz by default — the same convention used elsewhere in
the pipeline. Override `fmax` if your ERSPs were computed with a
different ceiling.

Public API:
  extract_hg_time_series(ersp, hg_band=..., fmax=...) -> (n_time,) np.ndarray
  build_hg_feature_matrix(ersp_list, ...)              -> (n_samples, n_time) X_hg
  save_sample_hg_png(ersp, path, ...)                  -> write sparkline PNG
  render_hg_sparkline(ax, hg_vec, ...)                 -> draw onto an axis
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ============================================================
# Defaults
# ============================================================
HG_BAND_DEFAULT = (70.0, 150.0)
FMAX_DEFAULT    = 500.0


# ============================================================
# Core: HG band extraction
# ============================================================
def extract_hg_time_series(
    ersp: np.ndarray,
    *,
    hg_band: Tuple[float, float] = HG_BAND_DEFAULT,
    fmax: float = FMAX_DEFAULT,
) -> np.ndarray:
    """
    Collapse the freq axis of an ERSP to a 1D high-gamma time series.

    Parameters
    ----------
    ersp : (n_freq, n_time) ERSP power (dB) matrix.
    hg_band : (lo_hz, hi_hz) frequency band to integrate over. Default
              (70, 150) Hz.
    fmax : maximum frequency of the ERSP axis. Default 500 Hz. The freq
           axis is reconstructed as linspace(0, fmax, n_freq).

    Returns
    -------
    hg : (n_time,) float32 array — mean ERSP across HG freq rows.
    """
    ersp = np.asarray(ersp, dtype=np.float32)
    n_freq, n_time = ersp.shape
    freqs = np.linspace(0.0, float(fmax), n_freq)
    lo, hi = float(hg_band[0]), float(hg_band[1])
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        # Fallback: middle 20% of the freq axis. Shouldn't happen for
        # sane settings (e.g. (70, 150) Hz with fmax=500 Hz), but keeps
        # the function safe.
        mask = np.zeros(n_freq, dtype=bool)
        mask[int(0.2 * n_freq):int(0.4 * n_freq)] = True
    return ersp[mask, :].mean(axis=0).astype(np.float32)


def build_hg_feature_matrix(
    ersp_list: Sequence[np.ndarray],
    *,
    hg_band: Tuple[float, float] = HG_BAND_DEFAULT,
    fmax: float = FMAX_DEFAULT,
    verbose: bool = True,
) -> np.ndarray:
    """
    Build the HG feature matrix X_hg of shape (n_samples, n_time).

    Each row is the HG band time series of the corresponding sample's
    ERSP.
    """
    n = len(ersp_list)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    _, n_time = np.asarray(ersp_list[0]).shape
    X = np.zeros((n, n_time), dtype=np.float32)
    for i, ersp in enumerate(ersp_list):
        X[i] = extract_hg_time_series(ersp, hg_band=hg_band, fmax=fmax)
    if verbose:
        print(f"[build_hg_feature_matrix] X_hg.shape={X.shape}  "
              f"hg_band={hg_band} Hz  fmax={fmax} Hz")
    return X


# ============================================================
# Per-sample / per-cluster sparkline render
# ============================================================
def render_hg_sparkline(
    ax,
    hg_vec,
    *,
    color: str = "#cc0033",
    show_axes: bool = False,
    ylim: Optional[Tuple[float, float]] = (-6.5, 6.5),  # match cfg.hg_vmin/vmax + the ERSP_clean cmap scale
    zero_line: bool = True,
):
    """
    Draw a 1D HG time series onto an existing matplotlib axis.

    Style: thin red line, optional baseline at 0, no axes/labels by
    default. Same minimalist look as the ERSP_clean PNGs so the chip
    grid stays visually consistent across feature sets.
    """
    hg = np.asarray(hg_vec).ravel()
    x = np.arange(len(hg))

    if zero_line:
        ax.axhline(0, color="#888", lw=0.4, alpha=0.6, zorder=1)
    # Fill above/below zero with light tint so direction reads at a glance
    ax.fill_between(x, 0, hg, where=hg > 0, color="#cc0033", alpha=0.20, lw=0, zorder=2)
    ax.fill_between(x, 0, hg, where=hg < 0, color="#0033cc", alpha=0.20, lw=0, zorder=2)
    ax.plot(x, hg, color=color, lw=1.1, zorder=3)

    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlim(0, max(1, len(hg) - 1))
    if not show_axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)


def save_sample_hg_png(
    ersp,
    path,
    *,
    figsize=(3, 1.5),
    dpi: int = 100,
    hg_band: Tuple[float, float] = HG_BAND_DEFAULT,
    fmax: float = FMAX_DEFAULT,
    ylim: Optional[Tuple[float, float]] = (-6.5, 6.5),  # match render_hg_sparkline + cfg.hg_vmin/vmax
):
    """Write a single-sample HG-sparkline PNG (no axes, no title)."""
    import matplotlib.pyplot as plt
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    hg = extract_hg_time_series(ersp, hg_band=hg_band, fmax=fmax)
    fig, ax = plt.subplots(figsize=figsize)
    render_hg_sparkline(ax, hg, ylim=ylim)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
