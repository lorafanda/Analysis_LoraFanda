#!/usr/bin/env python
"""151_hfa_tn.py - the HFA trial rasters on the ERSP's TIME-NORMALISED axis, so a band at a given
bin of the cube can be found in the single trials. Read-only on the pipeline: 140's loading,
cleaning and reference are captured the way 150 does it; nothing in outputs/03_ERSP changes.

    python .\151_hfa_tn.py --patient PAT_3455
    python .\151_hfa_tn.py --patient PAT_3455 --cond audio picture
    python .\151_hfa_tn.py --patient PAT_3455 --cond audio --channels AD2 IMD8 TOD9
    python .\151_hfa_tn.py --patient PAT_3455 --sort post            (rows by response duration; default stim, as the HFA figure)

THE QUESTION (2026-10-15, Lora): what is the band at bins 230-240 of the audio and picture cubes?
The HFA figure runs in seconds from the onset, so a bin of the time-normalised cube has no fixed
place on it. Here every trial's high-gamma z course (the HFA figure's own: cfg.hg_band
Butterworth(4) + Hilbert + cfg.hg_smooth_ms boxcar, z against cfg.baseline_w) is warped exactly
as compute_ersp warps a trial: stimulus onset..offset onto 150 bins, offset..trial end onto 150
bins (the cube's 300), and the baseline onto 30 fixed bins at the left. The x axis is the cube's:
0 = onset, 50 = GO / offset, 100 = trial end; a cube bin b sits at b / 3 %. Under the raster: the
mean z over the trials, and the same channel's 70-150 Hz course from the real cube
(outputs/03_ERSP) when it exists - the two should line up.

Rows are the trials that reached the ERSP, numbered as on the HFA figure, sorted like it (by
stimulus duration) unless --sort says otherwise. No trial-rejection marks: this is for looking.
Writes outputs/_hfa_tn/<pid>/<cond>/<pid>_<cond>_WM_HFAtrialsTN_<ch>.png
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

_spec = importlib.util.spec_from_file_location("crosschannel150", os.path.join(HERE, "150_trial_crosschannel.py"))
cc = importlib.util.module_from_spec(_spec)
sys.modules["crosschannel150"] = cc
_spec.loader.exec_module(cc)                           # load_140, capture_patient, hg_envelope

OUT_ROOT = os.path.join("outputs", "_hfa_tn")
NB, NS, NP = 30, 150, 150                              # baseline (fixed), stimulus, post: the cube's 150 + 150


def warp_trial(t, z, off_s, te_s, b0):
    """One trial's z course onto the time-normalised grid, piecewise, as compute_ersp._warp_piece."""
    def piece(lo, hi, n):
        m = (t >= lo) & (t < hi)
        if hi <= lo or m.sum() < 2:
            return np.full(n, np.nan)
        u = np.linspace(lo, hi, n, endpoint=False)
        return np.interp(u, t[m], z[m], left=z[m][0], right=z[m][-1])
    return np.concatenate([piece(b0, 0.0, NB), piece(0.0, off_s, NS), piece(off_s, te_s, NP)])


def cube_hg_course(real_root, pid, cond, ch, band):
    p = os.path.join(real_root, pid, "LM", "ERSP_matrix", cond, f"{pid}_{cond}_WM_ERSP_{ch}_TN.npy")
    if not os.path.exists(p):
        return None
    A = np.load(p).astype(float)
    f = np.arange(A.shape[0]) * (1000.0 / 256.0)
    mains = np.zeros(f.size, bool)
    for h in np.arange(50.0, f.max() + 1, 50.0):
        mains |= np.abs(f - h) <= 1.1 * (1000.0 / 256.0)
    m = (f >= band[0]) & (f < band[1]) & ~mains
    return A[m].mean(0)


def draw(pid, cond, ch, M, labels, order, meanz, cube, vmin, vmax, png):
    n = M.shape[0]
    fig = plt.figure(figsize=(9, 7.5), dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1.3], hspace=0.08)
    ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharex=ax)
    x0, x1 = -100.0 * NB / NS * 0.5, 100.0                     # baseline block = NB bins at the ERSP's bin width
    im = ax.imshow(M[order], aspect="auto", cmap="bwr", vmin=vmin, vmax=vmax, origin="lower",
                   extent=[x0, x1, 0.5, n + 0.5], interpolation="nearest")
    ax.axvline(0, color="k", lw=1); ax.axvline(50, color="m", lw=1.2, ls=":")
    ax.set_yticks(np.arange(1, n + 1)); ax.set_yticklabels([str(int(labels[i])) for i in order], fontsize=5)
    ax.tick_params(axis="y", length=0, pad=1)
    ax.set_ylabel("trials, sorted as the HFA figure (bottom = first)", fontsize=8)
    ax.set_title(f"{pid} - {cond} - WM-ref HFA trials, TIME-NORMALISED: {ch}\n"
                 f"0 = onset, 50 = GO/offset, 100 = trial end (cube bin b at b/3 %); baseline block on the left", fontsize=9)
    plt.setp(ax.get_xticklabels(), visible=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.06); cb.set_label("high-gamma z-score", fontsize=8)
    xs = np.concatenate([np.linspace(x0, 0, NB, endpoint=False), np.linspace(0, 50, NS, endpoint=False), np.linspace(50, 100, NP, endpoint=False)])
    ax2.plot(xs, meanz, color="#1f4e9c", lw=1.2, label="mean z over trials (this raster)")
    ax2.axhline(0, color="k", lw=0.5); ax2.axvline(0, color="k", lw=1); ax2.axvline(50, color="m", lw=1.2, ls=":")
    ax2.set_ylabel("z", fontsize=8, color="#1f4e9c")
    if cube is not None:
        ax3 = ax2.twinx()
        ax3.plot(np.arange(cube.size) / 3.0, cube, color="#c1121f", lw=1.2, label="70-150 Hz dB course of the real cube")
        ax3.set_ylabel("dB (cube)", fontsize=8, color="#c1121f")
        ax3.axhline(0, color="#c1121f", lw=0.3)
        h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper left")
    else:
        ax2.legend(fontsize=7, frameon=False, loc="upper left")
    ax2.set_xlim(x0, x1); ax2.set_xlabel("normalised time (%)", fontsize=8)
    ax2.set_xticks([x0, 0, 25, 50, 75, 100]); ax2.set_xticklabels(["base", "0", "25", "50", "75", "100"], fontsize=8)
    fig.savefig(png, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True)
    ap.add_argument("--cond", nargs="*", default=None, help="conditions (default all)")
    ap.add_argument("--channels", nargs="*", default=None, help="channel names (default every cube channel)")
    ap.add_argument("--sort", default="stim", choices=["stim", "post", "none"], help="row order (default stim, as the HFA figure)")
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--stub", default=None, help="folder with edf_stub.py, for a machine without mne")
    a = ap.parse_args()

    if importlib.util.find_spec("mne") is None:
        if not a.stub:
            sys.exit("mne is not installed here; run on the server, or give --stub <folder with edf_stub.py>")
        sys.path.insert(0, os.path.abspath(a.stub))
        import edf_stub
        edf_stub.install()

    m = cc.load_140()
    cfg = m.cfg
    pid = a.patient
    print(f"[151] {pid}: 140's loading, cleaning and reference, read-only (writers off, scratch tree deleted)")
    blocks, row = cc.capture_patient(m, pid)
    patient_id = str(row.get("patient_id") or pid)
    real_root = os.path.join(cfg.outputs_root, cfg.ERSP_TREE)
    b0 = float(cfg.baseline_w[0]); b1 = float(cfg.baseline_w[1])
    want_ch = set(a.channels) if a.channels else None

    for blk in blocks:
        cond = blk["cond"]
        if a.cond and cond not in a.cond:
            continue
        fs, sig = blk["fs"], blk["sig"]
        on, off, te = blk["on"], blk["off"], blk["te"]
        n = len(on)
        out_dir = os.path.join(os.path.abspath(a.out), patient_id, cond)
        os.makedirs(out_dir, exist_ok=True)
        stim_s = (off - on) / fs; post_s = (te - off) / fs
        order = (np.argsort(stim_s, kind="stable") if a.sort == "stim" else
                 np.argsort(post_s, kind="stable") if a.sort == "post" else np.arange(n))
        chans = [(ci, blk["names"][ci]) for ci in blk["chan_idx"] if want_ch is None or blk["names"][ci] in want_ch]
        print(f"[151] {patient_id} | {cond}: {len(chans)} channels x {n} trials -> {out_dir}")
        pre = int(round(b0 * fs))
        for ci, ch in chans:
            env = cc.hg_envelope(sig[:, ci].astype(float), fs, cfg.hg_band, cfg.hg_smooth_ms)
            M = np.full((n, NB + NS + NP), np.nan)
            for t in range(n):
                s = max(0, int(on[t]) + pre); e = max(s + 1, min(int(te[t]), sig.shape[0]))
                seg = env[s:e]
                tr = (np.arange(e - s) + s - int(on[t])) / fs
                bm = (tr >= b0) & (tr < b1)
                if not bm.any():
                    bm = np.zeros(seg.size, bool); bm[:max(1, seg.size // 10)] = True
                mu = float(np.mean(seg[bm])); sd = float(np.std(seg[bm], ddof=1)) or 1.0
                M[t] = warp_trial(tr, (seg - mu) / sd, float(stim_s[t]), float((te[t] - on[t]) / fs), b0)
            meanz = np.nanmean(M, 0)
            cube = cube_hg_course(real_root, patient_id, cond, ch, cfg.hg_band)
            png = os.path.join(out_dir, f"{patient_id}_{cond}_WM_HFAtrialsTN_{ch}.png")
            draw(patient_id, cond, ch, M, blk["labels"], order, meanz, cube, cfg.hg_vmin, cfg.hg_vmax, png)
            sys.stdout.write("*"); sys.stdout.flush()
        print(" done")
    print(f"-> {os.path.abspath(a.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
