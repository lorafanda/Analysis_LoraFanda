#!/usr/bin/env python3
"""make_notch_explainer.py - why a stripe survives a notch that removed the line's power,
on a synthetic recording, drawn with the pipeline's own functions.

The stripe in a dB-vs-baseline ERSP is trial power over baseline power at one row. It
measures the line's TIME COURSE, not its power. A line that is louder during fixation
than during the trial makes a blue stripe; and the time course of a frequency band lives
in the PHASES of its spectrum, not the amplitudes. Spectrum interpolation as published
(Leske & Dalal 2019) replaces the amplitudes and keeps the phases - right for stationary
mains, wrong here: the band still rises and falls with the trial after its power is
gone. Random phase makes the band stationary and the stripe disappears.

Four panels on one synthetic recording (pink noise + a 100 Hz line 3x stronger in the
0.8 s before each onset, onsets every 6 s):
    A  the recording around one onset, with the 95-105 Hz envelope
    B  the spectrum around 100 Hz: recorded, phase kept, random phase - the last two
       are indistinguishable HERE, which is the whole point
    C  the band's trial-averaged envelope: the phase-kept band still steps down at
       the onset; the random-phase band is flat
    D  the ERSP row at 100 Hz, trial-averaged, dB vs the pre-onset baseline: the stripe
Every number on the figure is computed. Writes a PNG and a JSON of the numbers into
outputs/_status_png/, for the KISS tab.

    python make_notch_explainer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch, butter, filtfilt, hilbert

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))
import lf_ersp as fe                                  # noqa: E402

OUT = ROOT / "outputs" / "_status_png"
INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"
RED, GREEN, BLUE = "#c1121f", "#1b7837", "#4a6fa5"
FS, T, F0 = 1000.0, 300.0, 100.0
BASE_W, TRIAL_S, PERIOD = (-0.6, -0.1), 3.0, 6.0


def synth(seed=21):
    t = np.arange(int(T * FS)) / FS
    rng = np.random.default_rng(seed)
    w = rng.normal(size=len(t)); f = np.fft.rfftfreq(len(t), 1 / FS); f[0] = f[1]
    noise = np.fft.irfft(np.fft.rfft(w) / np.sqrt(f), len(t)) * 20
    onsets = np.arange(3.0, T - 4, PERIOD)
    env = np.ones(len(t))
    for o in onsets:
        env[(t >= o - 0.8) & (t < o)] = 3.0
    line = 30.0 * env * np.sin(2 * np.pi * F0 * t + 0.3)
    return t, noise, line, env, onsets


def band_env(x, lo=95.0, hi=105.0):
    b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return np.abs(hilbert(filtfilt(b, a, x)))


def ersp_row(x, onsets):
    """The pipeline's spectrogram (128-sample Hann, nfft 256), the row nearest F0,
    dB relative to the pre-onset baseline, averaged over trials on a common axis."""
    f, tt, S = spectrogram(x, fs=FS, window="hann", nperseg=128, noverlap=96, nfft=256)
    row = 10 * np.log10(S[int(np.argmin(np.abs(f - F0)))] + 1e-30)
    rel = np.arange(-1.0, TRIAL_S, 0.05)
    trials = []
    for o in onsets:
        base = row[(tt >= o + BASE_W[0]) & (tt < o + BASE_W[1])].mean()
        trials.append(np.interp(o + rel, tt, row) - base)
    M = np.mean(trials, axis=0)
    stripe = float(M[(rel >= 0) & (rel < TRIAL_S)].mean())
    return rel, M, stripe


def trial_avg(x, onsets, rel):
    t = np.arange(len(x)) / FS
    return np.mean([np.interp(o + rel, t, x) for o in onsets], axis=0)


def main() -> int:
    t, noise, line, env, onsets = synth()
    x = noise + line
    # the pipeline's own interpolation, both ways, on the same recording
    au_k, au_r = [], []
    y_keep = fe.notch_by_interpolation(x, FS, base=50.0, max_hz=480.0, freqs=[F0], audit=au_k,
                                       phase="keep", widen=False)
    y_rand = fe.notch_by_interpolation(x, FS, base=50.0, max_hz=480.0, freqs=[F0], audit=au_r,
                                       phase="random", widen=True)
    hw_k, hw_r = au_k[0]["hw_hz"], au_r[0]["hw_hz"]
    versions = [("as recorded", x, RED), (f"interpolated, phase kept (±{hw_k:.0f} Hz)", y_keep, "#e08214"),
                (f"interpolated, random phase (±{hw_r:.0f} Hz)", y_rand, BLUE)]
    rows = {name: ersp_row(sig, onsets) for name, sig, _ in versions}
    numbers = {name: round(rows[name][2], 2) for name, _, _ in versions}
    numbers["clean noise"] = round(ersp_row(noise, onsets)[2], 2)

    fig, ax = plt.subplots(2, 2, figsize=(13.0, 8.6), dpi=150)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.08, hspace=0.42, wspace=0.22)
    fig.suptitle("Why a stripe survives a notch that removed the line: the stripe is the line's time course, "
                 "and the time course lives in the phases", x=0.06, ha="left", fontsize=12.5, color=INK, y=0.975)
    fig.text(0.06, 0.935, "Synthetic recording: pink noise plus a 100 Hz line that is 3x stronger in the 0.8 s before "
             "each onset than during the trial, onsets every 6 s - EL048's pattern.\nBoth interpolations are the "
             "pipeline's own function (lf_ersp.notch_by_interpolation); the ERSP row is the pipeline's spectrogram.",
             fontsize=9.2, color=MUTED, va="top", linespacing=1.5)

    # A - the recording around one onset
    a = ax[0, 0]
    o = onsets[10]
    m = (t >= o - 1.5) & (t <= o + 3.2)
    a.plot(t[m] - o, x[m], color=GREY, lw=0.4, label="signal")
    a.plot(t[m] - o, band_env(x)[m], color=RED, lw=1.6, label="envelope of the 95-105 Hz band")
    a.axvspan(BASE_W[0], BASE_W[1], color=INK, alpha=0.10, lw=0)
    a.axvspan(0, TRIAL_S, color=GREEN, alpha=0.06, lw=0)
    a.axvline(0, color=INK, lw=1.0, ls="--")
    a.set_ylim(top=a.get_ylim()[1] * 1.18)                # room for the labels above the envelope
    a.text(BASE_W[0] + 0.02, 0.97, "baseline", fontsize=8, color=INK, va="top", transform=a.get_xaxis_transform())
    a.text(0.08, 0.97, "trial", fontsize=8, color=GREEN, va="top", transform=a.get_xaxis_transform())
    a.set_xlabel("time from onset (s)", fontsize=9); a.set_ylabel("a.u.", fontsize=9)
    a.set_title("A  ·  the recording: the line is louder during fixation than during the trial",
                fontsize=10, loc="left", color=INK)
    a.legend(fontsize=8, frameon=False, loc="lower right")
    a.tick_params(labelsize=8, colors=MUTED)

    # B - the spectrum
    b = ax[0, 1]
    for name, sig, col in versions:
        f, P = welch(sig, fs=FS, nperseg=4000)
        k = (f >= 80) & (f <= 120)
        b.plot(f[k], 10 * np.log10(P[k]), color=col, lw=1.4 if col != BLUE else 1.0,
               ls="-" if col != BLUE else "--", label=name)
    b.set_xlabel("frequency (Hz)", fontsize=9); b.set_ylabel("PSD (dB)", fontsize=9)
    b.set_title("B  ·  the spectrum: after interpolation the two versions look the same",
                fontsize=10, loc="left", color=INK)
    b.legend(fontsize=8, frameon=False, loc="upper right")
    b.tick_params(labelsize=8, colors=MUTED)

    # C - the band's trial-averaged envelope
    c = ax[1, 0]
    rel = np.arange(-1.0, TRIAL_S, 0.01)
    for name, sig, col in versions:
        c.plot(rel, trial_avg(band_env(sig), onsets, rel), color=col, lw=1.6, label=name)
    c.axvspan(BASE_W[0], BASE_W[1], color=INK, alpha=0.10, lw=0)
    c.axvline(0, color=INK, lw=1.0, ls="--")
    c.set_xlabel("time from onset (s)", fontsize=9); c.set_ylabel("95-105 Hz envelope, trial mean (a.u.)", fontsize=9)
    c.set_title("C  ·  the band's time course: phases kept, the step survives; random phase, it is flat",
                fontsize=10, loc="left", color=INK)
    c.legend(fontsize=8, frameon=False, loc="upper right")
    c.tick_params(labelsize=8, colors=MUTED)

    # D - the ERSP row
    d = ax[1, 1]
    for name, sig, col in versions:
        r, M, s = rows[name]
        d.plot(r, M, color=col, lw=1.6, label=f"{name}:  {s:+.1f} dB")
    d.axhline(0, color=GREY, lw=0.8)
    d.axvspan(BASE_W[0], BASE_W[1], color=INK, alpha=0.10, lw=0)
    d.axvline(0, color=INK, lw=1.0, ls="--")
    d.set_xlabel("time from onset (s)", fontsize=9); d.set_ylabel("100 Hz row, dB vs baseline", fontsize=9)
    d.set_title("D  ·  the ERSP row: this is the stripe (mean over the trial in the legend)",
                fontsize=10, loc="left", color=INK)
    d.legend(fontsize=8, frameon=False, loc="upper right")
    d.tick_params(labelsize=8, colors=MUTED)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "notch_stripe_explainer.png"
    fig.savefig(png, dpi=150, facecolor="white"); plt.close(fig)
    (OUT / "notch_stripe_explainer.json").write_text(json.dumps(dict(
        stripe_db=numbers, hw_hz=dict(phase_kept=hw_k, random_phase=hw_r), n_iter_random=au_r[0]["n_iter"],
        line_fixation_over_trial=3.0, onsets=len(onsets), fs=FS, seconds=T,
        note="stripe = mean over the trial of the 100 Hz row in dB vs the -0.6..-0.1 s baseline, "
             "trial-averaged; 0 = no stripe"), indent=2))
    print("stripe at the 100 Hz row (dB, 0 = none):", numbers)
    print(f"bands: phase kept ±{hw_k:.0f} Hz, random ±{hw_r:.0f} Hz after {au_r[0]['n_iter']} passes")
    print(f"-> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
