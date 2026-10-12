#!/usr/bin/env python3
"""
make_feature_sets_explainer.py - E17: what each feature set takes from one real cube.

One electrode, one condition - PAT_3455 OTD2, audio, the cube behind
outputs/03_ERSP/PAT_3455/LM/ERSP/audio/PAT_3455_audio_WM_ERSP_OTD2_TN.png - drawn as the
pipeline draws it (bwr, +-6 dB, 0-400 Hz, normalised time with the 50 % line). On top of
it, the cells each feature set averages. The cells come from the pipeline's own rules
(rows whose centre frequency falls in [lo, hi), time cut into equal blocks), so the
rectangles are the rows and bins the code uses, not nominal edges. Under each cube, the
feature vector as the clustering sees it, on the same colour scale.

Row 1: the sets in use - concat_hg, concat_rawds, concat_bands5, concat_bands5z, and the
       counters' 13-170 Hz set.
Row 2: the candidates of 2026-10-12 - high gamma at 150 / 100 / 60 / 30 time bins, and
       the band below plus high gamma (13-70 + 70-150 Hz) at 30 bins.

WHAT IT IS MEANT TO MAKE OBVIOUS

  * every set is an AVERAGE over rectangles of this picture; the sets differ only in where
    the rectangles are and how many there are.
  * the picture repeats for the other two conditions, stitched along time: the feature
    count is bands x bins x 3.
  * what the figure cannot show: the per-electrode unit-norm applied before every fit, and
    bands5z's z-score, which uses one mean and SD per band over the whole cohort and so
    cannot be computed from one electrode (its strip shows the un-scored values).

Rows are selected exactly as lf_features.downsample_ersp_to_bands does (centre frequency in
[lo, hi) on a 0-398.4375 Hz axis of 103 rows); time blocks as lf_counters.band_tensor does
(mean over 300 / bins consecutive bins). concat_rawds / bands5 downsample time with the
pooling routine instead (an anti-aliased resize); the cells are the same, the values can
differ in the third decimal.

    python make_feature_sets_explainer.py
        -> outputs/clustering/explainers/E17_feature_sets.png  (+ E17_feature_sets.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
from matplotlib.patches import Rectangle             # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from functions.lf_features import FREQ_BANDS_15_TO_400HZ as B15, FREQ_BANDS_5_TO_400HZ as B5   # noqa: E402

TREE = "03_ERSP"                                     # the one tree since 2026-09-30
PID, COND, CH = "PAT_3455", "audio", "OTD2"
CUBE = ROOT.parent / "01_FBM_Analysis" / "outputs" / TREE / PID / "LM" / "ERSP_matrix" / COND / f"{PID}_{COND}_WM_ERSP_{CH}_TN.npy"
OUT = ROOT / "outputs" / "clustering" / "explainers"
FMAX, N_FREQ, N_TIME = 398.4375, 103, 300
F = np.linspace(0.0, FMAX, N_FREQ)                  # the builders' axis: row centres
DF = F[1] - F[0]
VLIM = 6.0
HG = (70.0, 150.0)                                   # lf_concat.DEFAULT_HG_BAND


def rows_of(lo: float, hi: float) -> np.ndarray:
    return np.flatnonzero((F >= lo) & (F < hi))


def cells(A: np.ndarray, bands, bins: int) -> np.ndarray:
    """(len(bands), bins): mean over the band's rows, then over 300 / bins consecutive bins."""
    step = N_TIME // bins
    assert step * bins == N_TIME, f"{bins} does not divide {N_TIME}"
    out = np.empty((len(bands), bins), np.float32)
    for i, (lo, hi) in enumerate(bands):
        x = A[rows_of(lo, hi)].mean(0)
        out[i] = x.reshape(bins, step).mean(1)
    return out


SETS = [
    # name, bands, bins, note
    ("concat_hg", [HG], 300, "today's HG: 70-150 Hz, every bin"),
    ("concat_rawds", B15, 30, "15 bands x 30 bins"),
    ("concat_bands5", B5, 30, "5 bands x 30 bins"),
    ("concat_bands5z", B5, 30, "the same cells, then z-scored\nper band over the cohort"),
    ("counters 13-170", [b for b in B15 if b[0] >= 13 and b[1] <= 170], 30, "7 bands x 30 bins (muscle contacts out)"),
    ("hg150", [HG], 150, "candidate: HG, 2 bins -> 1"),
    ("hg100", [HG], 100, "candidate: HG, 3 bins -> 1"),
    ("hg60", [HG], 60, "candidate: HG, 5 bins -> 1"),
    ("hg30", [HG], 30, "candidate: HG, 10 bins -> 1"),
    ("low+hg30", [(13.0, 70.0), HG], 30, "candidate: 13-70 and 70-150 Hz, 30 bins"),
]


def draw(ax_cube, ax_strip, A, name, bands, bins, note):
    ext = [0, 100, F[0] - DF / 2, F[-1] + DF / 2]
    ax_cube.imshow(A, origin="lower", aspect="auto", extent=ext, cmap="bwr", vmin=-VLIM, vmax=VLIM, alpha=0.28)
    used = np.full_like(A, np.nan)
    for lo, hi in bands:
        r = rows_of(lo, hi)
        used[r] = A[r]
    ax_cube.imshow(used, origin="lower", aspect="auto", extent=ext, cmap="bwr", vmin=-VLIM, vmax=VLIM)
    ax_cube.axvline(50, color="k", lw=1.0)
    for lo, hi in bands:
        r = rows_of(lo, hi)
        y0, y1 = F[r[0]] - DF / 2, F[r[-1]] + DF / 2
        ax_cube.add_patch(Rectangle((0, y0), 100, y1 - y0, fill=False, ec="k", lw=1.1, zorder=5))
        if bins < 300:
            for k in range(1, bins):
                ax_cube.plot([100 * k / bins] * 2, [y0, y1], color="k", lw=0.25, alpha=0.6, zorder=4)
    ax_cube.set_xlim(0, 100)
    ax_cube.set_ylim(0, 400)
    ax_cube.set_yticks([0, 100, 200, 300, 400])
    ax_cube.set_xticks([0, 50, 100])
    ax_cube.tick_params(labelsize=7)
    n_feat = len(bands) * bins * 3
    ax_cube.set_title(f"{name}\n{len(bands)} band{'s' if len(bands) > 1 else ''} x {bins} bins x 3 conditions = {n_feat:,} features",
                      fontsize=8.2, loc="left")
    ax_cube.text(1, 392, note, fontsize=6.4, va="top", ha="left", color="#333",
                 bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), zorder=6)
    V = cells(A, bands, bins)
    ax_strip.imshow(V, origin="lower", aspect="auto", cmap="bwr", vmin=-VLIM, vmax=VLIM,
                    extent=[0, 100, 0, len(bands)], interpolation="nearest")
    ax_strip.set_xlim(0, 100)
    ax_strip.set_xticks([])
    if len(bands) <= 7:
        ax_strip.set_yticks(np.arange(len(bands)) + 0.5)
        ax_strip.set_yticklabels([f"{lo:g}-{hi:g}" for lo, hi in bands], fontsize=5.6)
    else:
        ax_strip.set_yticks([0, len(bands)])
        ax_strip.set_yticklabels([f"{bands[0][0]:g}", f"{bands[-1][1]:g}"], fontsize=5.6)
    ax_strip.set_title("as the algorithm sees it", fontsize=6.2, loc="left", pad=2, color="#333")
    return n_feat, V


def main() -> int:
    A = np.load(CUBE).astype(np.float32)
    assert A.shape == (N_FREQ, N_TIME), A.shape
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(19, 9.2), dpi=170)
    gs = fig.add_gridspec(4, 5, height_ratios=[3.6, 1.2, 3.6, 1.2], hspace=0.48, wspace=0.30,
                          left=0.045, right=0.985, top=0.865, bottom=0.075)
    meta = {"source_cube": str(CUBE).replace("\\", "/"), "patient": PID, "condition": COND, "contact": CH,
            "freq_axis": "linspace(0, 398.4375, 103); a row belongs to a band when its centre is in [lo, hi)",
            "time": "300 bins per condition; a time bin is the mean of 300 / bins consecutive bins", "sets": []}
    for j, (name, bands, bins, note) in enumerate(SETS):
        row = 0 if j < 5 else 2
        ax_cube = fig.add_subplot(gs[row, j % 5])
        ax_strip = fig.add_subplot(gs[row + 1, j % 5])
        n_feat, V = draw(ax_cube, ax_strip, A, name, bands, bins, note)
        meta["sets"].append({"name": name, "bands_hz": [list(b) for b in bands], "time_bins": bins,
                             "features": n_feat, "rows_per_band": [int(rows_of(lo, hi).size) for lo, hi in bands],
                             "strip_min_max_db": [float(V.min()), float(V.max())]})
        if j % 5 == 0:
            ax_cube.set_ylabel("frequency (Hz)", fontsize=7)
        if row == 2:
            ax_strip.set_xlabel("normalised time (%)   0-50 stimulus, 50-100 response", fontsize=7)
    fig.text(0.045, 0.960, f"E17  -  what each feature set takes from one real cube   ({PID} {CH}, {COND}; +-{VLIM:g} dB, white = baseline)",
             fontsize=13, weight="bold")
    fig.text(0.045, 0.928, "Row 1: the sets in use.  Row 2: the candidates of 2026-10-12.  Bright = the rows the set averages, faded = never used. "
                           "Each set is an average over its rectangles; the picture repeats for picture and reading, stitched along time.",
             fontsize=8.4, color="#333")
    fig.text(0.045, 0.018, "Not shown: every electrode is unit-normed over its whole feature vector before any fit; mains rows (50, 100, 150 ... Hz) are kept by every set; "
                           "row 0 (0 Hz) is never used. The strips of concat_rawds / bands5 are drawn with block means; the pipeline resizes time with an anti-aliased filter (same cells).",
             fontsize=7.2, color="#555")
    png = OUT / "E17_feature_sets.png"
    fig.savefig(png, dpi=200, facecolor="white")
    plt.close(fig)
    (OUT / "E17_feature_sets.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {png}")
    for s in meta["sets"]:
        print(f"  {s['name']:<16} {len(s['bands_hz'])} bands x {s['time_bins']:>3} bins -> {s['features']:>5} features   rows/band {s['rows_per_band']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
