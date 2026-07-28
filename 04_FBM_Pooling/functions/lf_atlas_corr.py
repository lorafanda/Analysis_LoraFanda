"""
lf_atlas_corr.py — ERSP x atlas-probability correlation maps (lab-notes Figs 7-9).

The question behind it: instead of hard-assigning each electrode to a network (which
fails ~30% of the time, see the attribution matrix), treat network membership as a
CONTINUOUS value and ask *which parts of the time-frequency plane covary with it*.

    Fig 7  the ERSP cube: (contacts x freq x time), three condition slabs [a|p|r]
    Fig 8  one bin: ERSP(t,f) across contacts  vs  P(atlas) across the same contacts
           -> one correlation coefficient
    Fig 9  repeat for every bin, replot r back onto the time-frequency plane

Atlas-agnostic on purpose: `sample_atlas_at_contacts` takes ANY NIfTI volume in MNI152
space, so the Neurosynth association-z maps work today and the Fedorenko / LanA
probabilistic language atlas drops in unchanged the moment it is available.

Statistics
----------
- `method='spearman'` (default) — rank-based, robust to the heavily zero-inflated,
  skewed atlas-probability distribution and to ERSP outliers. `'pearson'` available.
- Correlations are computed in CHUNKS over bins so the (n_contacts x 116 100) matrix
  never has to be rank-transformed all at once.
- p-values from the standard t transform, then **Benjamini-Hochberg FDR across all
  TF bins** — the multiple-comparison correction the notes flag as open.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# fsaverage/MNI305 -> MNI152 (same constant lf_pool uses for the Neurosynth sampling)
MNI305_TO_152 = np.array([
    [0.9975, -0.0073, 0.0176, -0.0429],
    [0.0146, 1.0009, -0.0024, 1.5496],
    [-0.0130, -0.0093, 0.9971, 1.1840],
    [0.0, 0.0, 0.0, 1.0],
])


def sample_atlas_at_contacts(xyz_mni305: np.ndarray, nii_path, *,
                             already_mni152: bool = False) -> np.ndarray:
    """Value of an MNI152 NIfTI at each contact. xyz_mni305: (n, 3) fsaverage coords.
    Returns (n,) float; NaN where the contact falls outside the volume."""
    import nibabel as nib
    img = nib.load(str(nii_path))
    vol = np.asarray(img.dataobj, dtype=float)
    inv = np.linalg.inv(img.affine)
    xyz = np.asarray(xyz_mni305, dtype=float)
    if not already_mni152:
        xyz = (np.c_[xyz, np.ones(len(xyz))] @ MNI305_TO_152.T)[:, :3]
    ijk = np.rint(np.c_[xyz, np.ones(len(xyz))] @ inv.T)[:, :3].astype(int)
    out = np.full(len(xyz), np.nan)
    shape = np.array(vol.shape[:3])
    ok = ((ijk >= 0).all(axis=1)) & ((ijk < shape).all(axis=1))
    idx = ijk[ok]
    out[ok] = vol[idx[:, 0], idx[:, 1], idx[:, 2]]
    return out


def _rank_cols(A: np.ndarray) -> np.ndarray:
    """Average-tie ranks down each column."""
    from scipy.stats import rankdata
    return rankdata(A, axis=0)


def tf_correlation_map(X2d: np.ndarray, values: np.ndarray, *,
                       method: str = "spearman", chunk: int = 8000
                       ) -> Tuple[np.ndarray, int]:
    """Correlate every column of X2d (n_contacts x n_bins) against `values` (n_contacts,).

    Rows with NaN in `values` are dropped. Returns (r per bin, n_used)."""
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(v)
    X2d = X2d[keep]
    v = v[keep]
    n = len(v)
    if n < 3:
        raise ValueError(f"only {n} contacts with a finite atlas value")
    if method == "spearman":
        v = _rank_cols(v.reshape(-1, 1)).ravel()
    vc = v - v.mean()
    vss = float(np.sqrt((vc ** 2).sum()))

    nb = X2d.shape[1]
    r = np.empty(nb, dtype=np.float32)
    for i in range(0, nb, chunk):
        B = np.asarray(X2d[:, i:i + chunk], dtype=np.float64)
        if method == "spearman":
            B = _rank_cols(B)
        Bc = B - B.mean(axis=0)
        denom = np.sqrt((Bc ** 2).sum(axis=0)) * vss
        with np.errstate(divide="ignore", invalid="ignore"):
            r[i:i + chunk] = np.where(denom > 0, (Bc * vc[:, None]).sum(axis=0) / denom, 0.0)
    return r, n


def r_to_p(r: np.ndarray, n: int) -> np.ndarray:
    """Two-sided p for a correlation, via the t transform."""
    from scipy import stats
    r = np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return 2 * stats.t.sf(np.abs(t), df=n - 2)


def fdr_bh(p: np.ndarray, q: float = 0.05) -> Tuple[np.ndarray, float]:
    """Benjamini-Hochberg. Returns (significant mask, p-threshold).
    Implemented here so the module doesn't need statsmodels."""
    p = np.asarray(p, dtype=float).ravel()
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh_line = q * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh_line
    if not passed.any():
        return np.zeros(m, dtype=bool), 0.0
    kmax = np.max(np.nonzero(passed)[0])
    p_thr = ranked[kmax]
    return (p <= p_thr), float(p_thr)


def plot_tf_corr_map(r_map: np.ndarray, *, sig_mask: Optional[np.ndarray] = None,
                     n_blocks: int = 3, block_labels: Sequence[str] = ("audio", "picture", "reading"),
                     fmax_hz: float = 500.0, title: str = "", out_png=None,
                     vlim: Optional[float] = None, dpi: int = 130,
                     cmap: str = "RdBu_r", zero_contour: bool = False,
                     cbar_label: str = "correlation with atlas value"):
    """r_map: (n_freq, n_time_total). Draws the correlation on the TF plane with the
    condition-block dividers, the mid-block GO-cue, and (optionally) an FDR contour.

    The scale is always symmetric about zero (-v..+v). With a SEQUENTIAL cmap
    (e.g. green->yellow) pass `zero_contour=True`: a sequential ramp cannot encode sign
    on its own, so the r=0 line is drawn to show where the correlation flips direction."""
    import matplotlib.pyplot as plt
    n_freq, n_time = r_map.shape
    nb = n_time // n_blocks
    v = float(vlim if vlim is not None else max(0.05, np.nanpercentile(np.abs(r_map), 99)))
    fig, ax = plt.subplots(figsize=(13, 4.2))
    im = ax.imshow(r_map, aspect="auto", origin="lower", cmap=cmap, vmin=-v, vmax=v,
                   extent=[0, n_time, 0, fmax_hz])
    xs = np.linspace(0, n_time, n_time); ys = np.linspace(0, fmax_hz, n_freq)
    if zero_contour:
        from scipy.ndimage import uniform_filter
        ax.contour(xs, ys, uniform_filter(r_map.astype(float), size=9), levels=[0.0],
                   colors="#333333", linewidths=0.5, alpha=0.5)
    if sig_mask is not None and sig_mask.any():
        ax.contour(xs, ys, sig_mask.astype(float), levels=[0.5], colors="k", linewidths=0.7)
    for b in range(n_blocks):
        x0 = b * nb
        if b:
            ax.axvline(x0, color="k", lw=1.6)
        ax.axvline(x0 + nb / 2, color="0.35", lw=1.0, ls="--")   # GO-cue
        ax.text(x0 + nb / 2, fmax_hz * 0.94, block_labels[b], ha="center", fontsize=10)
    ax.set_xticks([b * nb + nb / 2 for b in range(n_blocks)])
    ax.set_xticklabels([f"{l}\n(0-100% of trial)" for l in block_labels], fontsize=8)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=cbar_label)
    plt.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    return fig
