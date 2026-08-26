#!/usr/bin/env python3
"""
make_bands_figure.py - what the three spectral feature sets do to the same ERSP.

One real electrode, four representations, so the cost of each step is visible rather
than argued:

    full resolution   129 freq x 900 time, the ERSP as computed
    concat_rawds      15 bands x 90, the published feature set
    concat_bands5     5 bands x 90, each a union of contiguous 15-band edges
    concat_bands5z    the same five, each z-scored to equal weight

THE FOURTH PANEL IS NOT ON A dB SCALE and has its own colour bar. z-scoring per band
leaves standard deviations within a band, not decibels, and putting it on the shared dB
bar would be the kind of quiet unit change this whole figure exists to make visible.

    python make_bands_figure.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))
import lf_concat as CC                                            # noqa: E402
from lf_features import FREQ_BANDS_15_TO_400HZ, FREQ_BANDS_5_TO_400HZ  # noqa: E402

SCRATCH = Path(r"C:\Users\fanda\AppData\Local\Temp\claude"
               r"\S--HumanNeuronLab-ANALYSIS-FLM-Analysis-LoraFanda"
               r"\b2b76878-a2dc-444b-8806-1d2b9386c369\scratchpad")
OUT = ROOT / "outputs" / "clustering" / "explainers" / "E8_band_schemes.png"
INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"
RED, GREEN = "#c1121f", "#1b7837"
CONDS = ("audio", "picture", "reading")
FMAX = 500.0


def band_ticks(edges):
    """Row centres and labels for a banded panel."""
    return ([i + 0.5 for i in range(len(edges))],
            [f"{int(lo)}-{int(hi)}" for lo, hi in edges])


def panel(ax, img, title, *, vlim, cmap="bwr", yt=None, ylab=None, freq_axis=None):
    n_r, n_c = img.shape
    im = ax.imshow(img, aspect="auto", origin="lower", cmap=cmap,
                   vmin=-vlim, vmax=vlim, interpolation="nearest",
                   extent=[0, n_c, 0, n_r])
    for b in (1, 2):                                  # condition seams
        ax.axvline(b * n_c / 3, color="k", lw=1.4, zorder=3)
    for b in range(3):                                # GO cue, mid-block
        ax.axvline((b + 0.5) * n_c / 3, color="#4a4f55", lw=0.9, ls=(0, (4, 3)),
                   zorder=3)
    ax.set_xticks([(b + 0.5) * n_c / 3 for b in range(3)])
    ax.set_xticklabels(CONDS, fontsize=8.4, color=INK)
    if yt is not None:
        ax.set_yticks(yt[0]); ax.set_yticklabels(yt[1], fontsize=7.0, color=MUTED)
    elif freq_axis is not None:
        hz = [1, 20, 70, 170, 270, 400]
        ax.set_yticks([h / FMAX * n_r for h in hz])
        ax.set_yticklabels([str(h) for h in hz], fontsize=7.0, color=MUTED)
    if ylab:
        ax.set_ylabel(ylab, fontsize=8.6, color=INK)
    ax.set_title(title, fontsize=10.4, loc="left", color=INK, pad=5)
    for s in ax.spines.values():
        s.set_color(GREY)
    return im


def main() -> int:
    Xe = np.load(SCRATCH / "ersp_examples.npy")
    meta = json.loads((SCRATCH / "ersp_examples.json").read_text())
    i = meta["labels"].index("strong high gamma")
    ersp, who = Xe[i], meta["who"][i]

    # every representation is built by the REAL builders, from this one electrode, so
    # the figure cannot drift from what the pipeline actually produces
    one = ersp[None, ...]
    x15 = CC.concat_rawds_features(one, n_blocks=3, fmax_hz=FMAX)[0].reshape(15, 90)
    x5 = CC.concat_bands5_features(one, n_blocks=3, fmax_hz=FMAX)[0].reshape(5, 90)
    # bands5z is a COHORT-level rescale - a single electrode has no cohort, so the
    # per-band mean and sd come from the full 1693-electrode matrix
    X5all = np.load(SCRATCH / "X5.npy").astype(float)
    x5z = x5.copy()
    for b in range(5):
        blk = X5all[:, b * 90:(b + 1) * 90]
        x5z[b] = (x5[b] - blk.mean()) / max(blk.std(), 1e-12)

    v_raw = float(np.percentile(np.abs(ersp), 99))
    v_b = float(np.percentile(np.abs(x15), 99.5))
    v_z = float(np.percentile(np.abs(x5z), 99.5))

    # A SECOND ELECTRODE, because one is not enough to show what z-scoring does.
    # The first is the strongest HIGH-GAMMA responder, so equalising the bands can only
    # flatter it; the second is low-frequency dominant.
    #
    # NOT a reversal, which is what this comment claimed before the ratios were printed
    # and read. z-scoring divides every band by the cohort's spread in it, and the low
    # band has by far the largest spread, so EVERY electrode's emphasis shifts away from
    # low frequencies. What differs is BY HOW MUCH - and a transform that moves two
    # electrodes by different factors moves them relative to each other, which is what
    # re-arranges the partition.
    j = meta["labels"].index("strong low frequency")
    lo_ersp, lo_who = Xe[j], meta["who"][j]
    lo5 = CC.concat_bands5_features(lo_ersp[None, ...], n_blocks=3,
                                    fmax_hz=FMAX)[0].reshape(5, 90)
    lo5z = lo5.copy()
    for b in range(5):
        blk = X5all[:, b * 90:(b + 1) * 90]
        lo5z[b] = (lo5[b] - blk.mean()) / max(blk.std(), 1e-12)

    fig = plt.figure(figsize=(15.6, 13.6), dpi=170)
    gs = GridSpec(5, 24, figure=fig, hspace=0.72, wspace=1.6,
                  left=0.075, right=0.955, top=0.876, bottom=0.158)

    fig.text(0.032, 0.975, "What each spectral feature set does to the same ERSP",
             fontsize=20.5, color=INK, va="top")
    fig.text(0.032, 0.930,
             f"One electrode - {who}, the strongest high-gamma responder in the cohort - "
             f"through all four representations, built by the pipeline's own functions.\n"
             f"Each step down the figure throws something away. The question is whether "
             f"what it throws away was carrying the clustering, and the first three "
             f"panels are on ONE shared dB scale so that is answerable by eye.",
             fontsize=9.5, color=MUTED, va="top", linespacing=1.6)

    axA = fig.add_subplot(gs[0, :22])
    imA = panel(axA, ersp, "A  ·  full resolution  —  129 frequencies × 900 time bins  "
                           "(the ERSP as computed)",
                vlim=v_raw, freq_axis=True, ylab="Hz")
    cax = fig.add_subplot(gs[0, 23]); plt.colorbar(imA, cax=cax)
    cax.tick_params(labelsize=7); cax.set_ylabel("dB", fontsize=8, color=MUTED)

    axB = fig.add_subplot(gs[1, :22])
    panel(axB, x15, "B  ·  concat_rawds  —  15 bands × 90  (1350 features, the published "
                    "set)",
          vlim=v_b, yt=band_ticks(FREQ_BANDS_15_TO_400HZ), ylab="band")

    axC = fig.add_subplot(gs[2, :22])
    imC = panel(axC, x5, "C  ·  concat_bands5  —  5 bands × 90  (450 features, same "
                         "colour scale as B)",
                vlim=v_b, yt=band_ticks(FREQ_BANDS_5_TO_400HZ), ylab="band")
    cax2 = fig.add_subplot(gs[1:3, 23]); plt.colorbar(imC, cax=cax2)
    cax2.tick_params(labelsize=7); cax2.set_ylabel("dB", fontsize=8, color=MUTED)

    axD = fig.add_subplot(gs[3, :22])
    imD = panel(axD, x5z, "D  ·  concat_bands5z  —  the same five bands, each z-scored "
                          "to equal weight   (NOT dB — its own scale)",
                vlim=v_z, cmap="PuOr_r", yt=band_ticks(FREQ_BANDS_5_TO_400HZ),
                ylab="band")
    cax3 = fig.add_subplot(gs[3, 23]); plt.colorbar(imD, cax=cax3)
    cax3.tick_params(labelsize=7)
    cax3.set_ylabel("SD within band", fontsize=8, color=MUTED)

    # ── E: the same two steps on a low-frequency-dominant electrode ──────────
    axE1 = fig.add_subplot(gs[4, :10])
    panel(axE1, lo5, f"E1  \u00b7  {lo_who}  \u2014  concat_bands5   "
                     f"(low-frequency dominant)",
          vlim=v_b, yt=band_ticks(FREQ_BANDS_5_TO_400HZ), ylab="band")
    axE2 = fig.add_subplot(gs[4, 12:22])
    imE = panel(axE2, lo5z, "E2  \u00b7  the same electrode  \u2014  concat_bands5z",
                vlim=v_z, cmap="PuOr_r", yt=band_ticks(FREQ_BANDS_5_TO_400HZ))
    caxE = fig.add_subplot(gs[4, 23])
    plt.colorbar(imE, cax=caxE)
    caxE.tick_params(labelsize=7)
    caxE.set_ylabel("SD within band", fontsize=8, color=MUTED)

    # per-band share of the COHORT's sum of squares, which is what the clustering spends
    band = np.array([n.split("|")[1] for n in
                     CC.concat_feature_names("concat_bands5")])
    bands = list(dict.fromkeys(band))
    mu = X5all.mean(0)
    tot = ((X5all - mu) ** 2).sum()
    share = [100 * ((X5all[:, band == b] - mu[band == b]) ** 2).sum() / tot
             for b in bands]
    txt = ("SHARE OF THE COHORT'S TOTAL VARIANCE, which is what a Euclidean distance "
           "actually spends its budget on:\n   concat_bands5   "
           + "   ".join(f"{b} {s:.0f}%" for b, s in zip(bands, share))
           + "\n   concat_bands5z  " + "   ".join(f"{b} ~20%" for b in bands)
           + "\n\nD is not a prettier C. Equalising the bands changes WHICH ELECTRODES "
             "cluster together - raw against z-scored gives ARI 0.37 on the same "
             "electrodes at the same K, a bigger\nchange than swapping k-means for Ward "
             "or convex NMF (those agree at 0.25-0.36). It is the largest single lever "
             "in the feature definition, so it is a separate feature set.")
    fig.text(0.075, 0.132, txt, fontsize=8.6, color=INK, va="top", linespacing=1.55)

    def emph(x):
        """How much of this electrode sits in HG relative to the low band."""
        return float(np.abs(x[2]).mean() / max(np.abs(x[0]).mean(), 1e-12))

    r1, r2 = emph(x5), emph(x5z)
    r3, r4 = emph(lo5), emph(lo5z)
    lines = [
        "READ E AGAINST C. Ratio of |70-170 Hz| to |1-20 Hz|, before and after:",
        f"   {who:<16}  bands5 {r1:5.2f}  ->  bands5z {r2:5.2f}   "
        f"(x{r2 / max(r1, 1e-9):.2f})",
        f"   {lo_who:<16}  bands5 {r3:5.2f}  ->  bands5z {r4:5.2f}   "
        f"(x{r4 / max(r3, 1e-9):.2f})",
        "",
        "BOTH ratios rise, and that is the point rather than an exception to it. The low "
        "band has the widest spread across the cohort, so dividing each band by its own",
        "spread shifts EVERY electrode's emphasis away from low frequencies. What "
        f"differs is by how much - x{r2 / max(r1, 1e-9):.2f} against "
        f"x{r4 / max(r3, 1e-9):.2f} here - and a transform that moves two electrodes by",
        "different factors moves them relative to EACH OTHER. That is what re-arranges "
        "the partition, and it is why this is a separate feature set and not a display "
        "choice.",
    ]
    fig.text(0.075, 0.052, "\n".join(lines), fontsize=8.6, color=INK, va="top",
             linespacing=1.55)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"electrode {who}")
    print(f"  full {ersp.shape}  ->  15-band {x15.shape}  ->  5-band {x5.shape}")
    print(f"  dB range: full +/-{v_raw:.2f}, banded +/-{v_b:.2f}; "
          f"z-scored +/-{v_z:.2f} SD")
    print(f"  variance share bands5: " + ", ".join(f"{b} {s:.1f}%"
                                                   for b, s in zip(bands, share)))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
