#!/usr/bin/env python3
"""
make_preprocessing_figures.py - the stage-01 chain, drawn, for the analysis_status site.

Every number and every parameter here was read out of the code that actually runs, not
from memory:

    functions/config.py            nperseg 128, nfft 256, noverlap 108,
                                   baseline_w (-0.6,-0.1), baseline_calc_w (-0.5,-0.1),
                                   proportions (0, .5, .5), n_time_bins 300, fmax 500
    functions/lf_ersp.py           _to_khz_resampled -> 1000 Hz, _spectro, compute_ersp
    functions/lf_trials.py         collect_trials filters
    140 cell 11                    the order: reref -> notch -> trials -> ERSP -> save
    02_.../functions/lf_dataset.py thr_pos 2.2 / min_prop_pos 0.02,
                                   thr_neg -3.0 / min_prop_neg 0.04
    02_.../functions/lf_concat.py  build_concat_dataset, concat_hg/rawds features

Cohort numbers are the v4 rebuild of 2026-08-26 (27 patients).
Real cubes are used wherever a real cube can make the point.

    python make_preprocessing_figures.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "outputs" / "04_ersp_LM_RAWONLY"
OUT = ROOT / "outputs" / "preprocessing_docs"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"
BLUE, RED, GREEN, ORANGE, PURPLE = "#4a6fa5", "#c1121f", "#1b7837", "#e08214", "#5b2c83"
MONO = {"family": "DejaVu Sans Mono"}
FS_DS, NPERSEG, NOVERLAP, NFFT = 1000.0, 128, 108, 256
EX = RAW / "EL033" / "LM" / "ERSP_matrix" / "audio" / "EL033_audio_WM_ERSP_aH_L11_TN.npy"
EXH = RAW / "EL033" / "LM" / "ERSP_halves" / "audio"


def arrow(ax, x0, y0, x1, y1, col=MUTED, lw=1.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, lw=lw, color=col, zorder=6))


def box(ax, x, y, w, h, title, sub, col, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                fc=fc, ec=col, lw=2.0, zorder=4))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=9.6,
            color=col, zorder=5, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.36, sub, ha="center", va="top", fontsize=7.4,
            color=MUTED, zorder=5, linespacing=1.45)


def header(fig, title, lines, y=0.985):
    fig.text(0.035, y, title, fontsize=19, color=INK, va="top")
    fig.text(0.035, y - 0.037, "\n".join(lines), fontsize=8.8, color=MUTED,
             va="top", linespacing=1.6)


def cube():
    return np.load(EX)


# ═══════════════════════════ P1 · the whole chain ═══════════════════════════
def p1():
    fig = plt.figure(figsize=(16.5, 6.6), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 17); ax.set_ylim(0, 7); ax.axis("off")
    header(fig, "Stage 01 — from raw sEEG to a clustering feature matrix",
           ["Nine steps. The first seven happen in notebook 140 and produce one cube per "
            "electrode per condition; the last two happen in 02_FBM_Clustering and turn "
            "those cubes into the matrix the algorithms see.",
            "Everything below was read from the code that runs, and the cohort numbers "
            "are the v4 rebuild of 2026-08-26."], y=0.975)

    steps = [
        ("1  LOAD", "signals + channel\nnames + fs", BLUE),
        ("2  DROP", "non-neural,\nmicro, grid", BLUE),
        ("3  RE-REFERENCE", "white matter\n(or grid CAR)", PURPLE),
        ("4  NOTCH", "mains harmonics,\nonly where real", PURPLE),
        ("5  TRIALS", "prep0 TSV\n+ filters", ORANGE),
        ("6  ERSP", "spectrogram,\nbaseline, warp", RED),
        ("7  SAVE", "cube 129×300\n+ odd/even halves", RED),
        ("8  GATE", "high-activity\nin ≥1 condition", GREEN),
        ("9  FEATURES", "concat 3 conditions\n→ hg / rawds", GREEN),
    ]
    w, gap, y = 1.55, 0.28, 3.5
    for i, (t, s, c) in enumerate(steps):
        x = 0.45 + i * (w + gap)
        box(ax, x, y, w, 1.35, t, s, c)
        if i < len(steps) - 1:
            arrow(ax, x + w + 0.03, y + 0.67, x + w + gap - 0.03, y + 0.67)

    ax.text(0.45, 3.15, "notebook 140  ·  one cube per electrode per condition",
            fontsize=9, color=MUTED)
    ax.plot([0.4, 12.6], [3.05, 3.05], color=MUTED, lw=1.2)
    ax.plot([12.9, 16.6], [3.05, 3.05], color=GREEN, lw=1.2)
    ax.text(12.9, 3.15, "02_FBM_Clustering", fontsize=9, color=GREEN)

    ax.text(0.45, 2.55, "WHAT SURVIVES EACH NARROWING  (v4, 27 patients)",
            fontsize=10.4, color=INK)
    rows = [("cubes written to ERSP_matrix", "9,774 files", MUTED),
            ("rows in the dataset cache", "9,342", MUTED),
            ("unique contacts", "3,296", INK),
            ("…with all THREE conditions present", "2,959", INK),
            ("…and high-activity in ≥1 condition", "1,693", GREEN),
            ("dropped: missing a condition", "149", MUTED),
            ("dropped: no high-activity condition", "1,266", MUTED),
            ("dropped: subdural grid (PAT_3415)", "192", MUTED),
            ("dropped: excluded patient EL044", "124 rows", MUTED)]
    for i, (k, v, c) in enumerate(rows):
        yy = 2.20 - i * 0.235
        ax.text(0.55, yy, k, fontsize=8.6, color=c)
        ax.text(5.1, yy, v, fontsize=8.6, color=c, ha="right", **MONO)

    ax.text(6.1, 2.20, "The gate is the big one: it removes 1,266 of 2,959 electrodes, "
            "43%.", fontsize=9.2, color=INK)
    ax.text(6.1, 1.94, "Of the 1,693 that survive, high-activity holds in\n"
                       "1 condition for 828,  2 for 465,  3 for 400.",
            fontsize=9, color=MUTED, va="top", linespacing=1.5)
    ax.text(6.1, 1.36, "An electrode enters on ONE condition and then contributes its "
                       "full three-condition\nprofile — the gate is per-electrode, not "
                       "per-condition.", fontsize=9, color=INK, va="top", linespacing=1.5)
    ax.text(6.1, 0.80, "Steps 3 and 4 change the SIGNAL. Step 5 changes which trials.\n"
                       "Step 8 changes which electrodes. Nothing after step 7 re-reads "
                       "the raw data.",
            fontsize=9, color=PURPLE, va="top", linespacing=1.5)
    p = OUT / "P1_pipeline_overview.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return p


# ═══════════════════════ P2 · re-reference + notch ═══════════════════════
def p2():
    rng = np.random.default_rng(3)
    fs, n = 1000.0, 2000
    t = np.arange(n) / fs
    drift = 40 * np.sin(2 * np.pi * 0.7 * t)                  # shared, non-neural
    mains = 18 * np.sin(2 * np.pi * 50 * t) + 7 * np.sin(2 * np.pi * 100 * t)
    resp = 22 * np.exp(-((t - 1.0) ** 2) / 0.004) * np.sin(2 * np.pi * 95 * t)
    wm = drift + mains + rng.normal(0, 4, n)                  # a WM contact: no response
    gm = drift + mains + resp + rng.normal(0, 4, n)           # grey matter: response

    fig = plt.figure(figsize=(16.0, 8.2), dpi=170)
    gs = fig.add_gridspec(2, 3, hspace=0.44, wspace=0.24,
                          left=0.055, right=0.98, top=0.70, bottom=0.07)
    header(fig, "Steps 3 – 4 · Re-referencing and the notch",
           ["Two operations that change the SIGNAL ITSELF, before a single trial is cut. "
            "Both run once per patient, on the whole recording.",
            "Re-referencing subtracts what every contact on a shaft shares. Whatever is "
            "common — drift, movement, mains pickup, the reference electrode's own "
            "activity — is not local brain activity, so subtracting it",
            "leaves what is. The white-matter contacts are used as that common signal "
            "because WM has no local task response of its own to remove."])

    a = fig.add_subplot(gs[0, 0])
    a.plot(t, gm, color=INK, lw=0.7); a.plot(t, wm - 130, color=ORANGE, lw=0.7)
    a.text(0.02, 0.95, "grey-matter contact", transform=a.transAxes, fontsize=8.4, color=INK)
    a.text(0.02, 0.10, "white-matter contact", transform=a.transAxes, fontsize=8.4, color=ORANGE)
    a.set_title("A · before — both carry the same drift and mains",
                fontsize=9.6, loc="left", color=INK)

    b = fig.add_subplot(gs[0, 1])
    b.plot(t, gm - wm, color=GREEN, lw=0.7)
    b.set_title("B · after WM re-reference — the shared part is gone",
                fontsize=9.6, loc="left", color=GREEN)
    b.text(0.02, 0.93, "grey − white", transform=b.transAxes, fontsize=8.4, color=GREEN)

    from scipy.signal import welch
    c = fig.add_subplot(gs[0, 2])
    for sig, col, lab in ((gm, INK, "before"), (gm - wm, GREEN, "after reref")):
        f, P = welch(sig, fs=fs, nperseg=512)
        c.semilogy(f, P, color=col, lw=1.1, label=lab)
    for h in (50, 100, 150, 200):
        c.axvline(h, color=RED, ls=":", lw=0.9)
    c.set_xlim(0, 250); c.legend(fontsize=7.6, frameon=False)
    c.set_xlabel("Hz", fontsize=8.4)
    c.set_title("C · the mains harmonics that the notch targets",
                fontsize=9.6, loc="left", color=INK)

    d = fig.add_subplot(gs[1, :])
    d.axis("off")
    d.text(0, 0.94, "WHICH SCHEME EACH PATIENT GETS  —  decided in 140, not by hand",
           fontsize=10.6, color=INK, transform=d.transAxes)
    rows = [
        ("depth electrodes with usable WM contacts", "apply_wm_reref  →  reref tag \"WM\"",
         "the normal case. WM contacts are the reference and are then EXCLUDED from the "
         "analysis — they cannot be clustered against themselves.", GREEN),
        ("grid patient, no WM contacts, listed in GRID_CAR_PATIENTS",
         "apply_grid_car  →  each array CAR'd separately",
         "a common average within each grid, so one array's noise does not leak into "
         "another's.", BLUE),
        ("grid patient not listed", "no re-referencing",
         "left raw rather than referenced against something inappropriate.", ORANGE),
        ("MicroEPI", "apply_wm_reref_selective (macros only)",
         "micros keep their own anchor if the preset names one, otherwise stay raw.",
         PURPLE)]
    for i, (when, what, why, col) in enumerate(rows):
        y = 0.74 - i * 0.215
        d.text(0.005, y, when, fontsize=8.8, color=col, transform=d.transAxes,
               fontweight="bold")
        d.text(0.30, y, what, fontsize=8.4, color=INK, transform=d.transAxes, **MONO)
        d.text(0.005, y - 0.075, why, fontsize=8.4, color=MUTED, transform=d.transAxes)

    d.text(0.005, -0.10, "THE NOTCH IS ADAPTIVE, not blanket. notch_mains_harmonics walks "
           "50 Hz and its harmonics and only notches one where a real peak is detected "
           "(z > 3 above the local PSD),", fontsize=8.8, color=INK, transform=d.transAxes)
    d.text(0.005, -0.175, "with Q set from how sharp that peak is. A harmonic that is not "
           "there is left alone, so no signal is removed on the assumption that noise "
           "must be present.", fontsize=8.8, color=INK, transform=d.transAxes)
    p = OUT / "P2_signal_conditioning.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return p


# ═══════════════════════ P3 · one trial to one ERSP ═══════════════════════
def p3():
    A = cube()
    rng = np.random.default_rng(11)
    fs, dur = 1000.0, 3.0
    t = np.arange(-0.6, dur, 1 / fs)
    sig = rng.normal(0, 1, t.size)
    sig += 2.2 * np.exp(-((t - 0.35) ** 2) / 0.03) * np.sin(2 * np.pi * 95 * t)
    sig += 0.8 * np.sin(2 * np.pi * 9 * t)

    fig = plt.figure(figsize=(16.0, 8.6), dpi=170)
    gs = fig.add_gridspec(1, 4, wspace=0.30, left=0.05, right=0.985,
                          top=0.74, bottom=0.17)
    header(fig, "Step 6 · How one electrode becomes one ERSP",
           ["Done once per electrode per condition, over every surviving trial. The "
            "signal is first resampled to 1000 Hz (_to_khz_resampled) so every patient "
            "lands on the same frequency grid regardless of their recording rate.",
            "Spectrogram: Hann window, nperseg 128, noverlap 108, nfft 256. That gives "
            "129 frequency bins from 0 to 500 Hz and a new time bin every 20 ms. The "
            "power is turned into dB with 10·log10.",
            "dB is then taken RELATIVE TO BASELINE: the mean over −0.5 to −0.1 s before "
            "onset is subtracted from every time bin, per frequency. So 0 dB means "
            "'no different from rest', not 'no power'."])

    a = fig.add_subplot(gs[0])
    a.plot(t, sig, color=INK, lw=0.5)
    a.axvspan(-0.5, -0.1, color=ORANGE, alpha=0.25, lw=0)
    a.axvline(0, color=RED, lw=1.4)
    _lo, _hi = a.get_ylim()
    a.text(-0.3, _lo * 0.80, "baseline\n−0.5 to −0.1 s", ha="center", va="bottom",
           fontsize=7.6, color=ORANGE)
    a.text(0.05, _hi * 0.90, "stimulus onset", fontsize=7.6, color=RED)
    a.set_xlabel("time (s)", fontsize=8.4)
    a.set_title("A · one trial, one electrode\nepoch starts −0.6 s before onset",
                fontsize=9.4, loc="left", color=INK)
    a.tick_params(labelsize=7.4, colors=MUTED)
    a.spines[["top", "right"]].set_visible(False)

    from scipy.signal import spectrogram
    f, tt, S = spectrogram(sig, fs=fs, window="hann", nperseg=NPERSEG,
                           noverlap=NOVERLAP, nfft=NFFT, scaling="density", mode="psd")
    Sdb = 10 * np.log10(np.maximum(S, 1e-20))
    b = fig.add_subplot(gs[1])
    b.imshow(Sdb, aspect="auto", origin="lower", cmap="magma",
             extent=[tt[0] - 0.6, tt[-1] - 0.6, f[0], f[-1]])
    b.set_ylim(0, 200); b.set_xlabel("time (s)", fontsize=8.4)
    b.set_ylabel("Hz", fontsize=8.4)
    b.set_title("B · spectrogram, raw dB\n129 bins 0–500 Hz, one every 20 ms",
                fontsize=9.4, loc="left", color=INK)
    b.tick_params(labelsize=7.4, colors=MUTED)

    bl = (tt - 0.6 >= -0.5) & (tt - 0.6 < -0.1)
    Srel = Sdb - Sdb[:, bl].mean(1, keepdims=True)
    c = fig.add_subplot(gs[2])
    c.imshow(Srel, aspect="auto", origin="lower", cmap="bwr", vmin=-6, vmax=6,
             extent=[tt[0] - 0.6, tt[-1] - 0.6, f[0], f[-1]])
    c.set_ylim(0, 200); c.axvline(0, color=INK, lw=1.0)
    c.set_xlabel("time (s)", fontsize=8.4)
    c.set_title("C · minus the baseline → dB re rest\nred = more than rest, blue = less",
                fontsize=9.4, loc="left", color=RED)
    c.tick_params(labelsize=7.4, colors=MUTED)

    d = fig.add_subplot(gs[3])
    im = d.imshow(A, aspect="auto", origin="lower", cmap="bwr", vmin=-6, vmax=6,
                  extent=[0, 300, 0, 500])
    d.axvline(150, color=INK, lw=1.4)
    d.text(75, 470, "stimulus", ha="center", fontsize=8, color=INK)
    d.text(225, 470, "post", ha="center", fontsize=8, color=INK)
    d.set_xlabel("time bin (0–299)", fontsize=8.4); d.set_ylabel("Hz", fontsize=8.4)
    d.set_title("D · averaged over trials, time-normalised\nA REAL cube: EL033 aH_L11, audio",
                fontsize=9.4, loc="left", color=GREEN)
    d.tick_params(labelsize=7.4, colors=MUTED)
    fig.colorbar(im, ax=d, fraction=0.04, pad=0.02, label="dB re baseline")

    fig.text(0.05, 0.075, "The saved cube is D: 129 frequencies × 300 time bins, one file "
             "per electrode per condition. NaNs left by the warp are filled from nearest "
             "neighbours (fill_nans_nearest) — the halves are NOT filled, because "
             "imputing the same bins in both halves would inflate their correlation.",
             fontsize=8.8, color=MUTED)
    p = OUT / "P3_one_trial_to_ersp.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return p


# ═══════════════════════ P4 · time normalisation ═══════════════════════
def p4():
    fig = plt.figure(figsize=(16.0, 8.0), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 8); ax.axis("off")
    header(fig, "Step 6b · Time normalisation — why every cube is 300 bins wide",
           ["Trials are not the same length. A patient answers one picture in 0.8 s and "
             "the next in 2.4 s. Averaging those on a clock axis smears the response, "
             "because 'one second after onset' is mid-stimulus in one",
            "trial and long past the answer in another. So each trial is WARPED onto a "
            "proportional axis before averaging: 0–100% of the stimulus, then 0–100% of "
            "the post-stimulus period.",
            "config: proportions = (0.0, 0.50, 0.50) and n_time_bins = 300. The first "
            "number is the pre-stimulus share and it is ZERO — the baseline is used to "
            "compute dB but is not kept in the output."])

    for k, (dur_s, dur_p, col, lab) in enumerate([
            (0.8, 1.1, BLUE, "a fast trial"), (2.4, 1.9, ORANGE, "a slow trial")]):
        y = 5.3 - k * 1.15
        tot = dur_s + dur_p
        ax.plot([1.0, 1.0 + 4.6 * dur_s / 3.5], [y, y], color=col, lw=9,
                solid_capstyle="butt")
        ax.plot([1.0 + 4.6 * dur_s / 3.5, 1.0 + 4.6 * tot / 3.5], [y, y], color=col,
                lw=9, alpha=0.4, solid_capstyle="butt")
        ax.text(0.9, y, lab, ha="right", va="center", fontsize=8.6, color=col)
        ax.text(1.0 + 4.6 * dur_s / 3.5 / 2, y - 0.30, f"stimulus {dur_s}s",
                ha="center", fontsize=7.4, color=col)
        ax.text(1.0 + 4.6 * (dur_s + dur_p / 2) / 3.5, y - 0.30, f"post {dur_p}s",
                ha="center", fontsize=7.4, color=col)
        arrow(ax, 6.4, y, 7.5, y, col=col)
        ax.plot([7.9, 10.2], [y, y], color=col, lw=9, solid_capstyle="butt")
        ax.plot([10.2, 12.5], [y, y], color=col, lw=9, alpha=0.4, solid_capstyle="butt")

    ax.text(6.95, 6.05, "warp", ha="center", fontsize=8.6, color=MUTED)
    ax.text(9.05, 5.95, "150 bins", ha="center", fontsize=8.2, color=INK)
    ax.text(11.35, 5.95, "150 bins", ha="center", fontsize=8.2, color=INK)
    ax.plot([10.2, 10.2], [3.6, 5.85], color=INK, lw=1.4, ls="--")
    ax.text(10.2, 3.42, "bin 150 = the stimulus OFFSET, in every trial of every patient",
            ha="center", fontsize=8.8, color=INK)

    ax.text(0.55, 2.95, "WHAT THE THREE SEGMENT EDGES ARE", fontsize=10.6, color=INK)
    for i, (a_, b_, txt) in enumerate([
        ("onset", "sample", "the photodiode-detected stimulus onset"),
        ("offset", "sample_offsets", "onset + onset_duration — the stimulus ends"),
        ("end", "trial_end", "the end of the trial window; the post segment is warped to "
         "this")]):
        y = 2.60 - i * 0.30
        ax.text(0.7, y, a_, fontsize=9, color=RED, fontweight="bold")
        ax.text(1.7, y, b_, fontsize=8.6, color=INK, **MONO)
        ax.text(4.3, y, txt, fontsize=8.6, color=MUTED)

    ax.text(0.55, 1.45, "THE CONSEQUENCE WORTH REMEMBERING", fontsize=10.6, color=RED)
    ax.text(0.7, 1.12, "A time bin is a PROPORTION, not a millisecond. Bin 225 is "
            "'halfway through the post-stimulus period' — which is 0.55 s after offset "
            "in the fast trial above and 0.95 s in the slow one.",
            fontsize=9, color=INK)
    ax.text(0.7, 0.82, "So the same bin index means the same STAGE of the trial across "
            "patients, but not the same latency. Any statement about a latency in "
            "milliseconds has to go back to the trial table.",
            fontsize=9, color=INK)
    ax.text(0.7, 0.44, "And because the post segment is warped to trial_end, anything "
            "that moves trial_end rescales the whole second half of every cube — which "
            "is exactly what the fixation-cross correction did for EL033 and PAT_3965.",
            fontsize=9, color=PURPLE)
    p = OUT / "P4_time_normalisation.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return p


# ═══════════════════ P5 · trial and electrode selection ═══════════════════
def p5():
    A = cube()
    fig = plt.figure(figsize=(16.0, 8.4), dpi=170)
    gs = fig.add_gridspec(1, 3, wspace=0.26, left=0.05, right=0.98,
                          top=0.63, bottom=0.44)
    header(fig, "Steps 5 and 8 · Which trials, and which electrodes",
           ["Two independent narrowings. Step 5 decides which TRIALS go into an average "
            "(lf_trials.collect_trials); step 8 decides which ELECTRODES enter the "
            "clustering at all (lf_dataset + lf_concat).",
            "They are easy to confuse because both are called 'filtering', but they act "
            "on different things and at different points in the chain."])

    a = fig.add_subplot(gs[0]); a.axis("off")
    a.text(0, 1.0, "STEP 5 — trial filters, in order", fontsize=10.4, color=ORANGE,
           transform=a.transAxes)
    for i, (name, rule) in enumerate([
            ("resp_accuracy", "must be correct / valid / 1"),
            ("min_stim_s", "stimulus ≥ 0.5 s"),
            ("min_post_s", "post-stimulus ≥ 1.0 s"),
            ("max_post_s", "post-stimulus ≤ 5.0 s"),
            ("IQR outlier", "post-stimulus within Q1−1.5·IQR … Q3+1.5·IQR")]):
        y = 0.86 - i * 0.115
        a.text(0.02, y, name, fontsize=8.6, color=INK, transform=a.transAxes, **MONO)
        a.text(0.42, y, rule, fontsize=8.4, color=MUTED, transform=a.transAxes)
    a.text(0, 0.24, "post_s = trial_end − sample_offsets, so the last three all read "
           "trial_end. Duplicate prep0 tables are dropped by MD5 first, or every trial "
           "would be counted twice.",
           fontsize=8.4, color=INK, transform=a.transAxes, va="top")

    b = fig.add_subplot(gs[1])
    im = b.imshow(A, aspect="auto", origin="lower", cmap="bwr", vmin=-6, vmax=6,
                  extent=[0, 300, 0, 500])
    hot = A > 2.2
    b.contour(np.linspace(0, 300, A.shape[1]), np.linspace(0, 500, A.shape[0]),
              hot.astype(float), levels=[0.5], colors="k", linewidths=0.6)
    b.set_title(f"STEP 8 — the gate, on a real cube\nblack outline: the "
                f"{100*hot.mean():.1f}% of bins above +2.2 dB",
                fontsize=9.4, loc="left", color=GREEN)
    b.set_xlabel("time bin", fontsize=8.4); b.set_ylabel("Hz", fontsize=8.4)
    b.tick_params(labelsize=7.4, colors=MUTED)
    fig.colorbar(im, ax=b, fraction=0.04, pad=0.02)

    c = fig.add_subplot(gs[2]); c.axis("off")
    c.text(0, 1.0, "STEP 8 — the high-activity rule", fontsize=10.4, color=GREEN,
           transform=c.transAxes)
    c.text(0.02, 0.86, "prop_above_pos ≥ 0.02      (fraction of the 129×300 plane\n"
                       "                             above +2.2 dB)",
           fontsize=8.5, color=INK, transform=c.transAxes, va="top", **MONO)
    c.text(0.02, 0.62, "        OR", fontsize=8.5, color=RED, transform=c.transAxes, **MONO)
    c.text(0.02, 0.54, "prop_below_neg ≥ 0.04      (below −3.0 dB)",
           fontsize=8.5, color=INK, transform=c.transAxes, va="top", **MONO)
    c.text(0, 0.34, "computed over the FULL 0–500 Hz cube, per condition. An electrode "
           "is kept if it holds in AT LEAST ONE of the three conditions — 2% of 38,700 "
           "bins is 774 bins.",
           fontsize=8.4, color=MUTED, transform=c.transAxes, va="top")
    c.text(0, 0.10, "v4: 1,693 of 2,959 pass.  828 on one condition, 465 on two, 400 on "
           "all three.", fontsize=8.6, color=GREEN, transform=c.transAxes, va="top")

    fig.text(0.05, 0.365, "THE THING TO KNOW ABOUT THIS GATE", fontsize=11, color=RED)
    fig.text(0.05, 0.335,
             "It is an AMPLITUDE test, so it selects loud electrodes rather than "
             "reproducible ones — and amplitude depends on how many trials went into the "
             "average. Fewer trials means a noisier mean, more bins past the threshold, "
             "and a better chance of passing.\n"
             "That is visible in v4: EL033 and PAT_3965 lost about 30% of their trials to "
             "the fixation-cross correction and both moved UP in gated count. The split-"
             "half cubes in ERSP_halves exist to replace this with a reproducibility test, "
             "which does not have that property.\n"
             "The gate is also the single largest narrowing in the whole pipeline: it "
             "removes 1,266 of 2,959 electrodes, 43% of everything that reaches it.",
             fontsize=8.9, color=MUTED, va="top", linespacing=1.65)
    p = OUT / "P5_trial_and_electrode_selection.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return p


# ═══════════════════════ P6 · cube to feature matrix ═══════════════════════
def p6():
    A = cube()
    fig = plt.figure(figsize=(16.0, 8.8), dpi=170)
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.26, left=0.055, right=0.98,
                          top=0.66, bottom=0.07)
    header(fig, "Step 9 · From three cubes to the matrix the algorithms see",
           ["Each surviving electrode has three cubes — audio, picture, reading. They "
            "are laid side by side along TIME into one 129 × 900 block, so the frequency "
            "axis is untouched and a single electrode is one row.",
            "That block is then reduced two different ways. Both are built from the same "
            "electrodes, so anything that differs between them is the representation and "
            "not the sample."])

    for i, (cond, col) in enumerate([("audio", BLUE), ("picture", ORANGE),
                                     ("reading", GREEN)]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(A, aspect="auto", origin="lower", cmap="bwr", vmin=-6, vmax=6,
                  extent=[0, 300, 0, 500])
        ax.set_title(f"{cond}   129 × 300", fontsize=9.2, loc="left", color=col)
        ax.set_xticks([])
        if i:                                  # keep the Hz scale on the first panel only
            ax.set_yticks([])
        else:
            ax.set_ylabel("Hz", fontsize=8.4); ax.tick_params(labelsize=7.4, colors=MUTED)

    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    ax.text(0, 1.02, "concatenated:  129 × 900", fontsize=10, color=INK,
            transform=ax.transAxes)
    ax.add_patch(Rectangle((0.005, 0.62), 0.36, 0.30, transform=ax.transAxes,
                           fc="#eef1f4", ec=INK, lw=1.4))
    for k, (lab, col) in enumerate([("audio", BLUE), ("picture", ORANGE), ("reading", GREEN)]):
        ax.text(0.005 + 0.06 + k * 0.12, 0.77, lab, fontsize=8.4, color=col,
                ha="center", transform=ax.transAxes)
        if k:
            ax.plot([0.005 + k * 0.12, 0.005 + k * 0.12], [0.62, 0.92], color=INK,
                    lw=1.0, transform=ax.transAxes)

    ax.text(0.42, 0.94, "concat_hg      →  (n, 900)", fontsize=9.6, color=PURPLE,
            transform=ax.transAxes, **MONO)
    ax.text(0.42, 0.84, "mean of the 70–150 Hz rows, per time bin. One number per bin, so "
            "900 features.\nHigh gamma is the band that tracks local firing, and the "
            "frequency axis collapses away.",
            fontsize=8.6, color=MUTED, transform=ax.transAxes, va="top", linespacing=1.5)

    ax.text(0.42, 0.56, "concat_rawds   →  (n, 1350)", fontsize=9.6, color=RED,
            transform=ax.transAxes, **MONO)
    ax.text(0.42, 0.46, "each condition block reduced to 15 frequency bands × 30 time "
            "bins, then re-stitched.\n15 × 3 × 30 = 1350. Downsampled PER BLOCK, never "
            "across the seam, so no feature\nmixes two conditions — and the grid matches "
            "the one stage-04 pooling uses.",
            fontsize=8.6, color=MUTED, transform=ax.transAxes, va="top", linespacing=1.5)

    ax.text(0, 0.30, "v4 cohort: 1,693 electrodes × 27 patients  →  concat_hg (1693, 900) "
            "and concat_rawds (1693, 1350)", fontsize=9.4, color=GREEN,
            transform=ax.transAxes)
    ax.text(0, 0.16, "These two matrices are what 240 / 241 / 242 read. Nothing after "
            "this point returns to the cubes, the trials or the raw signal.",
            fontsize=9.2, color=INK, transform=ax.transAxes)
    ax.text(0, 0.02, "concat_hg_all is the same construction with the step-8 gate lifted "
            "— a different cohort, so it can be read on its own but never compared "
            "electrode-for-electrode against the gated pair.",
            fontsize=8.8, color=PURPLE, transform=ax.transAxes)
    p = OUT / "P6_cube_to_features.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return p


if __name__ == "__main__":
    for fn in (p1, p2, p3, p4, p5, p6):
        print(f"  {fn().name}")
    print(f"\n-> {OUT}")
