#!/usr/bin/env python3
"""
make_microepi_ainp_figure.py - the Blackrock analog inputs around the task, per patient.

    python make_microepi_ainp_figure.py --patient G-01
    python make_microepi_ainp_figure.py --patient G-01 --files 272 273 274

WHY. The .ns6 micro stream carries three analog inputs (ainp1-3) after the 48 microwire
contacts. One of them is the microphone, so it is the only direct record of WHEN THE
PATIENT ACTUALLY SPOKE - the photodiode gives the stimulus, never the voice. This draws
all three so the microphone can be told from the others by eye.

READ ONLY. It opens the .ns6 files, reads three channels out of fifty-one, and writes a
PNG into outputs/microepi_ainp/. Nothing in DATARAW is touched.

WHICH FILES. The session folder is named for when RECORDING started, which can be a day
before the task; each file is ~5 minutes and ~0.9 GB. The file list comes from Lora (the
mtimes bracket the task start) because the NSx header's own t0 is 0.0034 s in every file
- it is per-file, not a wall clock, so it cannot place a file in the session.
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from neo.rawio import BlackrockRawIO
from scipy.signal import butter, sosfiltfilt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "microepi_ainp"
RAW = "//nasac-m2.unige.ch/m-HumanNeuronLab/DATARAW/MICROEPI"

# session folder and the files Lora identified as covering the task
SESSIONS = {
    # the file lists span the whole task, from the file holding its start to the end of the
    # photodiode train - the mtimes bracket it, see WHICH FILES above
    "G-01": dict(pat="PAT_5515", folder="MicroEPI-G-01/raw_blackrock/20241127-161902",
                 stem="20241127-161902", files=[273, 274, 275, 276, 277, 278, 279],
                 task="2024-11-28 15:52:08"),
    "G-02": dict(pat="PAT_5533", folder="MicroEPI-G-02/raw_blackrock/20250115-175330",
                 stem="20250115-175330", files=[222, 223, 224, 225, 226, 227, 228],
                 task="2025-01-16 13:56:00"),
    "G-03": dict(pat="PAT_6619", folder="MicroEPI-G-03/raw_blackrock/20250307-093441",
                 stem="20250307-093441", files=[55, 56, 57, 58, 59, 60],
                 task="2025-03-07 14:59:34"),
    "G-04": dict(pat="PAT_6704", folder="MicroEPI-G-04/raw_blackrock/20250402-170904",
                 stem="20250402-170904", files=[1289, 1290, 1291, 1292, 1293, 1294, 1295],
                 task="2025-04-07 14:55:53"),
    # G-05 is the retired .mat preset (it runs as PAT_6684 off the TRC), but the micro
    # recording exists; its config entry has flip=True, hence --flip here.
    # the photodiode only appears from -010 on (Lora), so -009 is left out
    "G-05": dict(pat="PAT_6684", folder="MicroEPI-G-05/raw_blackrock/20250618-145815",
                 stem="20250618-145815", files=[10, 11, 12, 13, 14, 15],
                 task="2025-06-18 15:39:19"),
    "G-06": dict(pat="PAT_6854", folder="MicroEPI-G-06/raw_blackrock/20260128-212210",
                 stem="20260128-212210", files=[229, 230, 231, 232, 233, 234],
                 task="2026-01-29 17:33:57"),
}
INK, MUTED = "#22272b", "#7c858e"
DS = 1000          # Hz to draw at; the envelope is computed at the full 30 kHz first


def read_ainp(path_noext, audio_chan=None, band=None):
    """(time, {label: trace at DS Hz}, {label: envelope at DS Hz}, fs). Three of 51 channels.

    `audio_chan` is band-passed to `band` before the envelope is taken. A microphone input
    sitting near full scale is dominated by whatever low-frequency component it carries, so
    the broadband envelope of it is flat whatever the patient says; the speech band is not.
    """
    r = BlackrockRawIO(filename=path_noext, nsx_to_load=6)
    r.parse_header()
    sig = r.header["signal_channels"]
    names = [str(x) for x in sig["name"]]
    fs = float(sig["sampling_rate"][0])
    idx = [i for i, n in enumerate(names) if "ainp" in n.lower()]
    if not idx:
        return None
    n = r.get_signal_size(0, 0)
    raw = r.get_analogsignal_chunk(block_index=0, seg_index=0, i_start=0, i_stop=n,
                                   channel_indexes=np.asarray(idx))
    raw = r.rescale_signal_raw_to_float(raw, dtype="float32",
                                        channel_indexes=np.asarray(idx))
    step = max(1, int(round(fs / DS)))
    win = max(1, int(round(0.025 * fs)))              # 25 ms, as the HG rasters use
    k = np.ones(win, dtype="float32") / win
    tr, en = {}, {}
    for j, i in enumerate(idx):
        x = raw[:, j]
        if audio_chan and names[i] == audio_chan and band:
            sos = butter(4, list(band), btype="band", fs=fs, output="sos")
            xf = sosfiltfilt(sos, x.astype("float64")).astype("float32")
        else:
            xf = x - np.median(x)
        # envelope at full rate, THEN decimate - decimating first would alias the
        # speech band away and the onset with it
        e = np.convolve(np.abs(xf), k, mode="same")
        tr[names[i]] = x[::step]
        en[names[i]] = e[::step]
    t = np.arange(len(next(iter(tr.values())))) / (fs / step)
    return t, tr, en, fs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True, choices=sorted(SESSIONS))
    ap.add_argument("--files", nargs="*", type=int, default=None)
    ap.add_argument("--zoom", nargs=2, type=float, default=None, metavar=("T0", "T1"),
                    help="seconds from the start of the first file shown; also marks the "
                         "ainp1 rising edges, so a voice channel can be read against the cue")
    ap.add_argument("--align", choices=("onset", "offset"), default="offset",
                    help="what t=0 is: the cue's onset, or its END (default) - the cue "
                         "going off is the go signal, so that is the response onset")
    ap.add_argument("--win", nargs=2, type=float, default=(4.0, 4.0), metavar=("PRE", "POST"),
                    help="raster window around t=0, in seconds (default 4 and 4)")
    ap.add_argument("--audio-chan", default=None,
                    help="channel carrying the microphone; its envelope is taken from the "
                         "speech band only (see --band)")
    ap.add_argument("--band", nargs=2, type=float, default=(300.0, 3000.0),
                    metavar=("LO", "HI"),
                    help="speech band in Hz, default 300-3000: formants without the F0 "
                         "range, where 50 Hz mains and its harmonics sit")
    ap.add_argument("--tmax", type=float, default=None,
                    help="drop everything after this many seconds from the start of the "
                         "first file shown. G-05 ends with non-biological spikes, so it is "
                         "cut midway through -015")
    ap.add_argument("--trig-chan", default=None,
                    help="channel carrying the photodiode; default ainp1 (G-06 uses ainp2)")
    ap.add_argument("--min-pulse", type=float, default=0.2,
                    help="a cue pulse must hold this long to count (s). A weak or fading "
                         "photodiode crosses a midpoint threshold on noise - G-05 gave 979 "
                         "edges in 25 min without this")
    ap.add_argument("--flip", action="store_true",
                    help="the photodiode is inverted for this patient (config 'flip'), so "
                         "the stimulus ONSET is a falling edge - G-05 and G-06")
    ap.add_argument("--raster", action="store_true",
                    help="one row per cue, envelope from -1 to +4 s, like the HG rasters - "
                         "a voice channel shows a band starting after 0")
    a = ap.parse_args()
    S = SESSIONS[a.patient]
    files = a.files if a.files else S["files"]
    folder = f"{RAW}/{S['folder']}"

    blocks = []
    for i in files:
        p = f"{folder}/{S['stem']}-{i:03d}"
        if not os.path.exists(p + ".ns6"):
            print(f"  {i:03d}: missing")
            continue
        print(f"  reading {S['stem']}-{i:03d}.ns6 ...", flush=True)
        got = read_ainp(p, audio_chan=a.audio_chan, band=tuple(a.band))
        if got is None:
            print("    no ainp channels")
            continue
        blocks.append((i, *got))
    if not blocks:
        raise SystemExit("nothing read")

    labels = sorted(blocks[0][2])
    fig, axes = plt.subplots(len(labels), 1, figsize=(15.5, 2.5 * len(labels) + 1.2),
                             dpi=170, sharex=True)
    axes = np.atleast_1d(axes)
    t_off, bounds = 0.0, []
    for i, t, tr, en, fs in blocks:
        for ax, lab in zip(axes, labels):
            ax.plot(t_off + t, tr[lab], lw=0.25, color="#9aa3ad", zorder=2)
            ax.plot(t_off + t, en[lab], lw=0.8, color="#c0392b", zorder=3)
        bounds.append((t_off, i))
        t_off += t[-1]

    # the cue train: the photodiode is high while the stimulus is up, so a rising edge is
    # its onset and the next falling edge its END - which is when the patient may answer.
    # Edges closer than 0.25 s are one event - the diode flickers on the on-ramp.
    lab0 = a.trig_chan or ("ainp1" if "ainp1" in labels else labels[0])
    if lab0 not in labels:
        raise SystemExit(f"no channel {lab0}; have {labels}")
    # Threshold and polarity are decided ONCE for the whole session, on the concatenated
    # trace. Per file they cannot be: a file with no stimuli in it has no two levels to
    # separate, so its threshold lands in noise and its duty cycle is arbitrary - that
    # flipped 3 of 6 files for G-03 and put the cue end in the wrong place in half the
    # trials. A pulse that runs past the end of its file is dropped either way.
    gx, gt, off = [], [], 0.0
    for i, t, tr, en, fs in blocks:
        gx.append(tr[lab0].astype("float64"))
        gt.append(off + t)
        off += t[-1]
    gx, gt = np.concatenate(gx), np.concatenate(gt)

    # The diode's RESTING LEVEL drifts between files - G-04 rests near 430 in some and 300
    # in others - so a threshold on the raw trace reads a whole file as "high" and the
    # pulses it finds are the gaps between stimuli (a 2.9 s median where the stimulus is
    # 1 s). Subtract a coarse resting level instead: the median of each 10 s block, which
    # is the rest level because the diode is down most of the time.
    blk = max(1, int(10 * DS))
    base = np.repeat(np.array([np.median(gx[j:j + blk]) for j in range(0, len(gx), blk)]),
                     blk)[:len(gx)]
    r = gx - base

    # polarity from the data, not from config 'flip' - that describes the .mat export and
    # disagrees with the raw channel (G-03 is flip=True yet reads straight). Whichever way
    # the stimulus pushes the diode off its resting level is the bigger excursion.
    p_hi, p_lo = float(np.percentile(r, 99)), float(np.percentile(r, 1))
    auto_inv = bool(a.flip) or abs(p_lo) > abs(p_hi)
    if auto_inv:
        r = -r
    amp = float(np.percentile(r, 99))
    b = (r > 0.5 * amp) if amp > 1e-9 else np.zeros(len(r), bool)
    w = max(1, int(round(a.min_pulse * DS)))
    if w > 1:
        # the box is centred, so a step reads as high only w/2 later - roll it back
        b = np.roll(np.convolve(b.astype("float32"), np.ones(w, "float32") / w, "same")
                    > 0.9, -(w // 2))
    up = np.flatnonzero(b[1:] & ~b[:-1]) + 1
    dn = np.flatnonzero(~b[1:] & b[:-1]) + 1
    cues = []                                   # (onset, end), s from the first file shown
    for s in up:
        j = int(np.searchsorted(dn, s))
        if j >= len(dn):                        # the pulse runs past the end of the data
            continue
        on, offt = float(gt[s]), float(gt[dn[j]])
        if not cues or on - cues[-1][0] > 0.25:
            cues.append((on, offt))
    if a.tmax is not None:
        cues = [c for c in cues if c[0] <= a.tmax]
    trig = [(c[1] if a.align == "offset" else c[0]) for c in cues]
    dur = float(np.median([o - n for n, o in cues])) if cues else 0.0

    for ax, lab in zip(axes, labels):
        for b, _ in bounds[1:]:
            ax.axvline(b, color=INK, lw=0.8, ls="--", alpha=0.5, zorder=4)
        ax.set_ylabel(lab, fontsize=10, color=INK)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if a.zoom is not None:
            for v in trig:
                if a.zoom[0] <= v <= a.zoom[1]:
                    ax.axvline(v, color="#1b6ca8", lw=0.7, alpha=0.75, zorder=1)
            ax.set_xlim(*a.zoom)
        elif a.tmax is not None:
            ax.set_xlim(0, a.tmax)
    for b, i in bounds:
        axes[0].text(b + 3, axes[0].get_ylim()[1], f"-{i:03d}", fontsize=7,
                     color=MUTED, va="top")
    axes[-1].set_xlabel("seconds from the start of the first file shown", fontsize=10)
    fig.suptitle(f"{a.patient} ({S['pat']}) · Blackrock analog inputs, "
                 f"files {files[0]}–{files[-1]} · task starts {S['task']}",
                 fontsize=12.5, color=INK, y=0.995)
    sub = ("grey = raw trace at 1 kHz · red = 25 ms envelope of |signal| computed at the "
           "full 30 kHz · dashed = file boundary")
    if a.audio_chan:
        sub += f" · {a.audio_chan} band-passed {a.band[0]:.0f}-{a.band[1]:.0f} Hz first"
    fig.text(0.5, 0.955, sub, fontsize=8.5, color=MUTED, ha="center")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_zoom{int(a.zoom[0])}-{int(a.zoom[1])}" if a.zoom is not None else ""
    png = OUT / f"ainp_{a.patient}_{files[0]}-{files[-1]}{tag}.png"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n-> {png}")

    # one time base for the whole window, so the cue-locked means can be taken
    cat_t, cat, off = [], {lab: [] for lab in labels}, 0.0
    for i, t, tr, en, fs in blocks:
        cat_t.append(off + t)
        for lab in labels:
            cat[lab].append(en[lab])
        off += t[-1]
    cat_t = np.concatenate(cat_t)
    cat = {lab: np.concatenate(v) for lab, v in cat.items()}
    if a.tmax is not None:                       # keep the stats off the end-of-run spikes
        m = cat_t <= a.tmax
        cat_t, cat = cat_t[m], {k: v[m] for k, v in cat.items()}

    how = "forced inverted (--flip)" if a.flip else (
        "auto-inverted" if auto_inv else "straight")
    print(f"   {len(cues)} cues on {lab0} ({how}), median {dur:.2f} s long; "
          f"t=0 is the cue {'end' if a.align == 'offset' else 'onset'}")
    for lab in labels:
        y = cat[lab]
        md = float(np.median(y))
        pk = float(np.percentile(y, 99))
        pre, post, n = 0.0, 0.0, 0
        for v in trig:
            p0, p1 = np.searchsorted(cat_t, [v - 1.0, v - 0.2])
            q0, q1 = np.searchsorted(cat_t, [v + 0.2, v + 1.5])
            if p1 > p0 and q1 > q0:
                pre += float(y[p0:p1].mean())
                post += float(y[q0:q1].mean())
                n += 1
        lock = f"post/pre {post/pre:5.2f} over {n} cues" if n and pre > 0 else "no cues"
        print(f"   {lab}: envelope median {md:.4g}, p99 {pk:.4g}, "
              f"ratio {pk/max(md,1e-12):.1f} | {lock}")

    if a.raster and cues:
        dt = float(np.median(np.diff(cat_t)))
        n_pre, n_post = int(round(a.win[0] / dt)), int(round(a.win[1] / dt))
        tt = np.arange(-n_pre, n_post) * dt
        nl = len(labels)
        fig2, ax2 = plt.subplots(2 * nl, 1, figsize=(7.6, 3.2 * nl + 1.0), dpi=170,
                                 sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1] * nl, hspace=0.15))
        for r_i, lab in enumerate(labels):
            y, rows = cat[lab], []
            for on, off in cues:
                c = int(np.searchsorted(cat_t, off if a.align == "offset" else on))
                i0, i1 = c - n_pre, c + n_post
                if i0 < 0 or i1 > len(y):
                    continue
                seg = y[i0:i1].astype("float64")
                # baseline is the second BEFORE the cue came on: with t=0 at the cue end,
                # the window from -1 to 0 is the stimulus itself
                b0, b1 = np.searchsorted(cat_t, [on - 1.0, on])
                base = float(y[b0:b1].mean()) if b1 > b0 else float(seg[:n_pre].mean())
                rows.append(seg - base)
            if not rows:
                continue
            M = np.asarray(rows)
            lim = float(np.percentile(np.abs(M), 99)) or 1.0
            top, bot = ax2[2 * r_i], ax2[2 * r_i + 1]
            top.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       extent=[tt[0], tt[-1], len(M), 0], interpolation="nearest")
            top.set_ylabel(f"{lab}\ncue", fontsize=9.5, color=INK)
            bot.plot(tt, M.mean(0), lw=1.1, color="#c0392b")
            bot.axhline(0, color=MUTED, lw=0.6)
            bot.set_ylabel("mean", fontsize=8)
            for ax in (top, bot):
                ax.axvline(0, color=INK, lw=1.0)
                if a.align == "offset" and dur > 0:   # where the cue came on
                    ax.axvline(-dur, color=INK, lw=0.8, ls=":", alpha=0.7)
                ax.tick_params(labelsize=8)
            for s in ("top", "right"):
                bot.spines[s].set_visible(False)
        ax2[-1].set_xlabel("seconds from the end of the cue" if a.align == "offset"
                           else "seconds from the cue", fontsize=9.5)
        fig2.suptitle(f"{a.patient} ({S['pat']}) · envelope per cue, baseline-subtracted · "
                      f"files {files[0]}–{files[-1]} · {len(cues)} cues"
                      + (f" · dotted = cue on ({dur:.1f} s)" if a.align == "offset" else ""),
                      fontsize=11.5, color=INK)
        png2 = OUT / f"ainp_{a.patient}_{files[0]}-{files[-1]}_raster.png"
        fig2.savefig(png2, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
        print(f"-> {png2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
