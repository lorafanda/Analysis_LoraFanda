#!/usr/bin/env python
"""150_trial_crosschannel.py - which trials look the same on EVERY channel: the cross-channel
correlation of the high-gamma envelope, per trial, for one patient. Read-only: 140's own
loading, cleaning and reference, and nothing is written to outputs/03_ERSP.

    python .\150_trial_crosschannel.py --patient PAT_3455
    python .\150_trial_crosschannel.py --patient PAT_3455 --k 3.0          (a softer mark)
    python .\150_trial_crosschannel.py --patient PAT_3455 --stub <folder with edf_stub.py>
                                                            (a machine without mne; the server needs no stub)

THE QUESTION (2026-10-15, Lora): a big spike that is propagated across channels has nothing to
do with the task, and the per-channel trial rejection cannot see that it is the same event
everywhere. Here every trial gets one number - how alike the channels are during it.

WHAT IS COMPUTED. Per channel and trial, the HG envelope z the HFA raster draws: cfg.hg_band
Butterworth(4) band-pass + Hilbert envelope + cfg.hg_smooth_ms boxcar, z against the trial's
own cfg.baseline_w, from the baseline start to the trial end (the filter runs over the whole
block rather than per segment: the same envelope away from the segment edges, a hundred
times faster). Channels are the ones that get a cube - neural, not bad-listed, not a
reference contact - and the trials are the ones that reached the ERSP, numbered as on the
HFA figure. Per trial, across those channels:
  r_mean   the mean of the channel-by-channel correlations of the z courses
           ("how much the channels agree during this trial")
  r_i      per channel, its correlation with the mean course of the other channels (the heatmap)
  peak_z   the largest value of the across-channel mean z course (how big the shared event is)
A trial is MARKED when r_mean > median + k * 1.4826 * MAD over the condition's trials
(k = --k, 3.5 by default: the same robust rule the ERSP's trial rejection uses). The task
response is shared by many channels on every trial, so it raises r_mean for all trials
alike and sets the median; a trial that stands above the others is one where the channels
agree more than the task makes them - a propagated event.

THE FIGURE, like 148's: the conditions side by side, rows = channels grouped by shaft,
columns = trials, colour = r_i (0 pale .. 1 dark); above each panel a strip with r_mean per
trial, the threshold as a dashed line, the marked trials red with a marker over the column.

HOW. 140 is imported as a module and its process_patient run with every writer off, its
output roots pointed at a temporary folder (deleted at the end); the block after reference
and notch, the channel list and the kept trials are captured on the way - the exact arrays
the cubes are made from. Writes outputs/_trial_crosschannel/<pid>/<pid>_trial_crosschannel.png
and one TSV per condition (trial, trial_label, r_mean, peak_z, threshold, marked).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from scipy.fft import next_fast_len                     # noqa: E402
from scipy.signal import butter, filtfilt, hilbert      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUT_ROOT = os.path.join("outputs", "_trial_crosschannel")


# ----------------------------------------------------------------------------- capture
def load_140():
    spec = importlib.util.spec_from_file_location("ersp140", os.path.join(HERE, "140_ersp_pipeline.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["ersp140"] = m
    spec.loader.exec_module(m)
    return m


def capture_patient(m, pid):
    """Run 140's process_patient read-only and return, per condition, the cleaned block and the
    trials: dict(cond, fs, sig, names, chan_idx, on, off, te, labels)."""
    tmp = tempfile.mkdtemp(prefix=f"crosschannel_{pid}_")
    m.run_root_ersp = tmp
    m.run_root_raw = tmp
    for flag in ("WRITE_MONTAGE", "WRITE_ERSP_PLOTS", "WRITE_HG_PLOTS", "WRITE_SIGNAL_PLOTS", "WRITE_CUBES",
                 "WRITE_HALVES", "WRITE_CLEAN_PNG", "EXPORT_TRIAL_SCORES", "WRITE_WM_NOREF"):
        setattr(m, flag, False)
    cap = dict(conds=None, all_trials=None, names=None, blocks=[])

    _collect = m.tr.collect_trials

    def collect_hook(*a, **kw):
        groups = _collect(*a, **kw)
        cap["conds"] = list(groups.keys())
        cap["all_trials"] = kw.get("all_out")          # filled in place by collect_trials: every trial and why it went
        return groups

    m.tr.collect_trials = collect_hook

    _notch = m.apply_notch_with_audit

    def notch_hook(signals, fs, patient_id, pid_raw, audit=None, names=None):
        if names is not None:
            cap["names"] = list(names)
        return _notch(signals, fs, patient_id, pid_raw, audit=audit, names=names)

    m.apply_notch_with_audit = notch_hook

    def ersp_hook(signals, fs, onsets, offsets, channel_idx, *, trial_ends=None, **kw):
        # the first call of a condition carries the block the cubes are made from; every call
        # names one channel that gets a cube (140 has already skipped bad / reference / non-neural)
        key = (id(signals), int(np.asarray(onsets)[0]), len(onsets))
        if not cap["blocks"] or cap["blocks"][-1]["key"] != key:
            cap["blocks"].append(dict(key=key, fs=float(fs), sig=np.asarray(signals, np.float32),
                                      on=np.asarray(onsets, int), off=np.asarray(offsets, int),
                                      te=np.asarray(trial_ends, int), chan_idx=[]))
        cap["blocks"][-1]["chan_idx"].append(int(channel_idx))
        return dict(avg_db=np.zeros((2, 2)), meta=dict(mode="TN"))

    m.fe.compute_ersp = ersp_hook

    try:
        row = m.process_patient(pid, False, True, False, False)   # no QC figures, the export path (dummied), no PSD, no micros
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if str(row.get("status", "")).lower().startswith("err"):
        raise RuntimeError(f"140 failed for {pid}: {row.get('error')}")
    if not cap["blocks"]:
        raise RuntimeError(f"{pid}: 140 produced no exported channel (status {row.get('status')})")
    if cap["conds"] is None or len(cap["conds"]) != len(cap["blocks"]):
        raise RuntimeError(f"{pid}: {len(cap['blocks'])} blocks for conditions {cap['conds']}")

    out = []
    for cond, blk in zip(cap["conds"], cap["blocks"]):
        names = cap["names"] or [f"ch{i}" for i in range(blk["sig"].shape[1])]
        n = len(blk["on"])
        labels = np.arange(1, n + 1)
        times = dict(on_s=np.full(n, np.nan), off_s=np.full(n, np.nan), tend_s=np.full(n, np.nan))
        rec = (cap["all_trials"] or {}).get(cond)
        if rec is not None and n:
            # 140's own numbering (the HFA figure, TrialScores): the position among the trials
            # whose span lies INSIDE the block - a removed trial that runs out of the block is
            # not counted - so the whole table goes into block coordinates first, as 140 does
            keep_a = np.asarray(rec["keep"], bool)
            on_a = np.asarray(rec["on"], float)
            te_a = np.asarray(rec["tend"], float)
            kept_on = on_a[keep_a]
            shift = (float(kept_on[0]) - float(blk["on"][0])) if kept_on.size else 0.0
            inb = (on_a - shift >= 0) & (te_a - shift <= blk["sig"].shape[0])
            if int((keep_a & inb).sum()) == n:
                labels = np.flatnonzero(keep_a[inb]) + 1      # the number the HFA figure prints beside the row
                sel = keep_a & inb                            # the kept trials on the recording clock, for bad_time_spans
                times = dict(on_s=on_a[sel] / blk["fs"], off_s=np.asarray(rec["off"], float)[sel] / blk["fs"],
                             tend_s=te_a[sel] / blk["fs"])
            else:
                print(f"  [{pid} | {cond}] trial tables disagree ({int((keep_a & inb).sum())} vs {n}); trials numbered 1..{n}")
        out.append(dict(cond=cond, fs=blk["fs"], sig=blk["sig"], names=names, chan_idx=blk["chan_idx"],
                        on=blk["on"], off=blk["off"], te=blk["te"], labels=labels, **times))
    return out, row


# ----------------------------------------------------------------------------- the numbers
def hg_envelope(x, fs, band, smooth_ms):
    nyq = 0.5 * fs
    b, a = butter(4, [max(1.0, band[0]) / nyq, min(band[1] / nyq, 0.999)], btype="bandpass")
    bp = filtfilt(b, a, x, method="gust")
    env = np.abs(hilbert(bp, N=next_fast_len(bp.size))[:bp.size])
    k = max(1, int(round(smooth_ms / 1000.0 * fs)))
    if k % 2 == 0:
        k += 1
    if k > 1:
        env = np.convolve(env, np.ones(k) / k, mode="same")
    return env


def trial_z_courses(block, cfg):
    """Per channel that gets a cube, per kept trial: the HG envelope z of the raster.
    Returns (chan_names, Z) with Z a list over trials of (n_channels x n_samples) arrays."""
    fs, sig = block["fs"], block["sig"]
    on, te = block["on"], block["te"]
    b0, b1 = float(cfg.baseline_w[0]), float(cfg.baseline_w[1])
    pre = int(round(b0 * fs))
    segs = []
    for t in range(len(on)):
        s = max(0, int(on[t]) + pre)
        e = max(s + 1, min(int(te[t]), sig.shape[0]))
        segs.append((s, e))
    chans = [block["names"][ci] for ci in block["chan_idx"]]
    t0 = np.array([(s - int(on[t])) / fs for t, (s, e) in enumerate(segs)])   # each course's first sample, s from the onset
    Z = [np.full((len(chans), e - s), np.nan) for s, e in segs]
    for row, ci in enumerate(block["chan_idx"]):
        env = hg_envelope(sig[:, ci].astype(float), fs, cfg.hg_band, cfg.hg_smooth_ms)
        for t, (s, e) in enumerate(segs):
            seg = env[s:e]
            t_rel = (np.arange(e - s) + s - int(on[t])) / fs
            bm = (t_rel >= b0) & (t_rel < b1)
            if not bm.any():
                bm = np.zeros(seg.size, bool); bm[:max(1, seg.size // 10)] = True
            mu = float(np.mean(seg[bm])); sd = float(np.std(seg[bm], ddof=1)) or 1.0
            Z[t][row] = (seg - mu) / sd
        sys.stdout.write("."); sys.stdout.flush()
    print()
    return chans, Z, t0


def crosschannel(Z, t0, fs):
    """Per trial: r_mean, r_i per channel, the peak of the across-channel mean z and when it is
    (s from the onset)."""
    n = len(Z)
    C = Z[0].shape[0]
    r_mean = np.full(n, np.nan)
    peak_z = np.full(n, np.nan)
    peak_t = np.full(n, np.nan)
    R = np.full((C, n), np.nan)
    for t, M in enumerate(Z):
        M = np.asarray(M, float)
        ok = np.isfinite(M).all(1)
        Mc = M - np.nanmean(M, 1, keepdims=True)
        sd = np.sqrt(np.nanmean(Mc ** 2, 1))
        Mn = Mc / (sd[:, None] + 1e-12)
        if ok.sum() >= 2:
            Cm = (Mn[ok] @ Mn[ok].T) / M.shape[1]
            c = int(ok.sum())
            r_mean[t] = (np.nansum(Cm) - np.trace(Cm)) / (c * (c - 1))
        mean_all = np.nanmean(M, 0)
        peak_z[t] = float(np.nanmax(mean_all))
        peak_t[t] = float(t0[t]) + float(np.nanargmax(mean_all)) / float(fs)
        if C >= 2:
            others = (np.nansum(M, 0)[None, :] - M) / (C - 1)        # mean of the other channels, per channel
            Oc = others - others.mean(1, keepdims=True)
            R[:, t] = (Mc * Oc).mean(1) / (sd * Oc.std(1) + 1e-12)
    return r_mean, peak_z, peak_t, R


def mark(r_mean, k):
    ok = np.isfinite(r_mean)
    if ok.sum() < 4:
        return np.zeros(len(r_mean), bool), np.nan
    med = float(np.median(r_mean[ok]))
    mad = float(np.median(np.abs(r_mean[ok] - med))) * 1.4826
    thr = med + float(k) * mad if mad > 1e-9 else np.inf
    return ok & (r_mean > thr), thr


# ----------------------------------------------------------------------------- the figure (148's layout)
def shaft_of(name: str) -> str:
    return re.sub(r"\d+$", "", re.sub(r"[-_]", "", str(name)).upper())


def contact_no(name: str) -> int:
    m = re.search(r"(\d+)$", str(name))
    return int(m.group(1)) if m else 0


def channel_order(names) -> list[str]:
    first = {}
    for n in names:
        first.setdefault(shaft_of(n), len(first))
    return sorted(set(names), key=lambda n: (first[shaft_of(n)], contact_no(n), str(n)))


def draw(pid, results, k, png):
    conds = [r["cond"] for r in results]
    chans = channel_order([c for r in results for c in r["chans"]])
    n_ch = len(chans)
    widths = [max(1, len(r["labels"])) for r in results]
    fig = plt.figure(figsize=(2.2 + 0.11 * sum(widths) + 0.6 * len(conds), 2.4 + 0.075 * n_ch), dpi=150)
    gs = fig.add_gridspec(2, len(conds), width_ratios=widths, height_ratios=[1.0, max(2.5, 0.075 * n_ch)],
                          wspace=0.06, hspace=0.04)
    im = None
    axes_hm = []
    for j, r in enumerate(results):
        n = len(r["labels"])
        x = np.arange(n)
        ax_s = fig.add_subplot(gs[0, j])
        ax = fig.add_subplot(gs[1, j])
        axes_hm.append(ax)
        # the strip: r_mean per trial, the threshold, the marked trials in red
        ax_s.bar(x, np.nan_to_num(r["r_mean"]), width=0.8,
                 color=np.where(r["marked"], "#c1121f", "#9a9a9a"), linewidth=0)
        if np.isfinite(r["thr"]):
            ax_s.axhline(r["thr"], color="#c1121f", lw=0.7, ls="--")
        ax_s.set_xlim(-0.5, n - 0.5)
        ax_s.set_ylim(0, 1)
        ax_s.set_xticks([])
        ax_s.set_yticks([0, 0.5, 1.0])
        ax_s.tick_params(labelsize=5.5, length=2, pad=1.5)
        if j:
            ax_s.set_yticklabels([])
        else:
            ax_s.set_ylabel("r_mean", fontsize=6.5)
        for sp in ("top", "right"):
            ax_s.spines[sp].set_visible(False)
        marked = [str(int(l)) for l, f in zip(r["labels"], r["marked"]) if f]
        ax_s.set_title(f"{r['cond']}  ·  marked (r_mean > median + {k:g} MAD = {r['thr']:.2f}): "
                       + (" ".join(marked) if marked else "none"), fontsize=8, pad=6, loc="left")
        # the heatmap: r_i per channel and trial
        Zm = np.full((n_ch, n), np.nan)
        for row, ch in enumerate(r["chans"]):
            Zm[chans.index(ch)] = r["R"][row]
        im = ax.imshow(Zm, aspect="auto", cmap="magma_r", vmin=0, vmax=1, interpolation="none")
        shafts = [shaft_of(c) for c in chans]
        starts = [i for i in range(n_ch) if i == 0 or shafts[i] != shafts[i - 1]]
        for i in starts[1:]:
            ax.axhline(i - 0.5, color="white", lw=0.8)
        if j == 0:
            ends = starts[1:] + [n_ch]
            ax.set_yticks([(a + b - 1) / 2 for a, b in zip(starts, ends)])
            ax.set_yticklabels([shafts[i] for i in starts], fontsize=6.5)
        else:
            ax.set_yticks([])
        step = 1 if n <= 60 else 2
        ax.set_xticks(x[::step])
        ax.set_xticklabels([str(int(v)) for v in r["labels"]][::step], fontsize=4.8, rotation=90)
        ax.tick_params(length=0, pad=1.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for t in np.flatnonzero(r["marked"]):
            ax.plot(t, -1.0, marker="v", ms=4, color="#c1121f", clip_on=False)
        ax.set_ylim(n_ch - 0.5, -0.5)
        ax.set_xlim(-0.5, n - 0.5)
    cb = fig.colorbar(im, ax=axes_hm, fraction=0.012, pad=0.01)
    cb.set_label("r  (channel vs the mean of the other channels)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.text(0.005, 0.995, f"{pid}  —  cross-channel correlation of the HG envelope z, per trial "
                           f"(the strip: r_mean = mean channel-pair correlation; dashed: median + {k:g} MAD)",
             fontsize=9, fontweight="bold", va="top")
    fig.savefig(png, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True)
    ap.add_argument("--k", type=float, default=3.5, help="mark r_mean > median + k MAD (default 3.5)")
    ap.add_argument("--out", default=OUT_ROOT, help=f"output root (default {OUT_ROOT}); never 03_ERSP")
    ap.add_argument("--stub", default=None, help="folder with edf_stub.py, for a machine without mne")
    a = ap.parse_args()

    if importlib.util.find_spec("mne") is None:
        if not a.stub:
            sys.exit("mne is not installed here; run on the server, or give --stub <folder with edf_stub.py>")
        sys.path.insert(0, os.path.abspath(a.stub))
        import edf_stub
        edf_stub.install()

    m = load_140()
    cfg = m.cfg
    pid = a.patient
    print(f"[150] {pid}: 140's loading, cleaning and reference, read-only (writers off, scratch tree deleted)")
    blocks, row = capture_patient(m, pid)
    patient_id = str(row.get("patient_id") or pid)
    out_dir = os.path.join(os.path.abspath(a.out), patient_id)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for blk in blocks:
        print(f"[150] {patient_id} | {blk['cond']}: {len(blk['chan_idx'])} channels x {len(blk['on'])} trials, "
              f"HG envelope z {cfg.hg_band[0]:g}-{cfg.hg_band[1]:g} Hz, {cfg.hg_smooth_ms} ms boxcar, "
              f"baseline {cfg.baseline_w}")
        chans, Z, t0 = trial_z_courses(blk, cfg)
        r_mean, peak_z, peak_t, R = crosschannel(Z, t0, blk["fs"])
        go_t = (blk["off"] - blk["on"]) / blk["fs"]                 # the stimulus offset = the GO cue, s from the onset
        marked, thr = mark(r_mean, a.k)
        results.append(dict(cond=blk["cond"], chans=chans, labels=blk["labels"], r_mean=r_mean, peak_z=peak_z,
                            R=R, marked=marked, thr=thr))
        df = pd.DataFrame(dict(patient=patient_id, condition=blk["cond"], trial=np.arange(len(r_mean)),
                               trial_label=blk["labels"], n_channels=len(chans),
                               onset_s=np.round(blk["on_s"], 3), offset_s=np.round(blk["off_s"], 3),   # recording clock,
                               tend_s=np.round(blk["tend_s"], 3),                                     # for bad_time_spans
                               r_mean=np.round(r_mean, 4), peak_z=np.round(peak_z, 3),
                               peak_t_s=np.round(peak_t, 3),                 # when the shared course peaks, s from the onset
                               peak_after_go_s=np.round(peak_t - go_t, 3),   # the same, s from the GO cue (negative = during the stimulus)
                               threshold=np.round(thr, 4), marked=marked.astype(int)))
        tsv = os.path.join(out_dir, f"{patient_id}_{blk['cond']}_trial_crosschannel.tsv")
        df.to_csv(tsv, sep="\t", index=False)
        med = float(np.nanmedian(r_mean))
        print(f"       r_mean median {med:.2f}, threshold {thr:.2f}; marked "
              f"{int(marked.sum())} of {len(r_mean)}: "
              + (" ".join(str(int(l)) for l, f in zip(blk["labels"], marked) if f) or "none"))
        for t in np.flatnonzero(marked):
            print(f"         trial {int(blk['labels'][t]):>3}: r_mean {r_mean[t]:.2f}, peak of the mean z {peak_z[t]:.1f}")
    png = os.path.join(out_dir, f"{patient_id}_trial_crosschannel.png")
    draw(patient_id, results, a.k, png)
    print(f"-> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
