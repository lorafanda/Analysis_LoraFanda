"""Step 2 - the Micromed macro analysis, with the LM preprocessing chain, from the TRC.

    python .\\20_mm_macro.py --patient G-05                    (LM chain: bad -> WM ref -> notch)
    python .\\20_mm_macro.py --patient G-05 --reref none       (unreferenced: the WM control run)
    python .\\20_mm_macro.py --patient G-05 --contrast-only    (skip the per-channel figures)
    python .\\20_mm_macro.py --patient G-05 --channels IMG1,IMG9

THE CHAIN IS THE LM CHAIN. Every preprocessing step here is a call into
01_FBM_Analysis/functions - not a reimplementation - in the order notebook 140 applies
them, so the two tasks are cleaned identically and a difference between an LM and an MM
figure can only come from the data:

    1. bad channels     suggest_bad_channels_by_psd + cfg.bad_channels_manual
    2. WM reference     wm_indices_for_patient -> apply_wm_reference_with_exclusions,
                        good WM contacts only; the WM contacts are then not analysed
    3. per block        block_window -> apply_notch_with_audit (per shaft, LM settings)
                        with plot_psd_overview before and after
    4. per channel      compute_ersp(mode="RT") and plot_hg_trials, with the ERSP's own
                        dropped-trial list handed to the HG figure

What is MM-specific is only the geometry of a trial: no time warping (mode="RT", every
trial is a fixed 3 s GO after a variable rest), baseline -1.0..-0.2 s, and the square
20/80 display. Those live in config_mm.py; everything about the room and the amplifier
comes from the LM config.

THE TWO REFERENCING MODES ARE DIFFERENT EXPERIMENTS. --reref wm is the analysis: the LM
chain, with the white-matter contacts consumed as the reference and excluded from the
figures, exactly as LM does. --reref none is the CONTROL: nothing subtracted, so the WM
contacts are analysed as ordinary channels and must sit near zero - if they carry the same
response as cortex, the response is not cortical. The two cannot be done in one run,
because a WM reference defines the WM contacts as zero by construction.

WHY THIS SCRIPT NEEDS A CLOCK OFFSET AND THE MICRO ONE DOES NOT. The cue is on the
Blackrock only - the Micromed was searched channel by channel and has no photodiode - so
cue times are carried across with align_macro_offset_s, measured by cross-correlating
micro against macro on the same shaft. Good to about +-0.35 s: fine for cutting a 6.2 s
trial, not for placing a spike against a cue.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "functions"))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "..", "01_FBM_Analysis")))
import config_mm as cfg                                        # noqa: E402
from lf_mm_io import read_trc_meta, pull_channels              # noqa: E402

INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"


# ---------------------------------------------------------------------------
# the LM functions, imported once and reported once if absent
# ---------------------------------------------------------------------------
def lm():
    """The stage-01 modules. mne is needed by lf_io_utils; say so plainly if missing."""
    try:
        from functions import config as lm_cfg
        from functions import lf_ersp as fe
        from functions import lf_io_utils as io
        return lm_cfg, fe, io
    except Exception as e:
        raise SystemExit(
            f"cannot import the LM functions ({type(e).__name__}: {e}).\n"
            f"  This script applies the LM preprocessing chain by calling it, so it needs\n"
            f"  the same environment notebook 140 runs in (mne is required).")


# ---------------------------------------------------------------------------
def load_trials(pat, pre):
    p = os.path.join(ROOT, "outputs", pat, f"{pre['pat_name']}_MM_trials.tsv")
    if not os.path.exists(p):
        raise SystemExit(f"no trial table at {p}\n"
                         f"  run:  python .\\10_mm_triggers.py --patient {pat}")
    import pandas as pd
    return pd.read_csv(p, sep="\t")


def trc_path(pre):
    t = pre["trc"]
    t = t[-1] if isinstance(t, list) else t
    root = pre["out_dir"].split("task_")[0] if "task_" in pre["out_dir"] else pre["out_dir"]
    for dirpath, _dn, fn in os.walk(root):
        if dirpath.count(os.sep) - root.count(os.sep) > 3:
            continue
        for f in fn:
            if f.lower() == t.lower():
                return os.path.join(dirpath, f)
    raise SystemExit(f"could not find {t} under {root}")


def load_trc(meta):
    """Every neural channel of the TRC at its native rate, (n_samples, n_channels)."""
    drop = re.compile(r"^(MKR|ECG|X\d|\.+)", re.I)
    idxs = [i for i, n in enumerate(meta["names"]) if n and not drop.match(n)]
    got, fs = pull_channels(meta, idxs, target_fs=None, dtype=meta["dtype"])
    X = np.stack([got[i] for i in idxs], axis=1).astype(np.float64)
    # Micromed stores unsigned 16-bit around a mid-scale ground; the LM loader sees the
    # same values through mne. Remove the DC offset per channel so nothing downstream sees
    # a 32768-unit pedestal as signal.
    X -= np.median(X, axis=0, keepdims=True)
    return X, [meta["names"][i] for i in idxs], fs


# ---------------------------------------------------------------------------
def plot_ersp_square(pat, ch, per, png):
    """Square panels, baseline 20% / GO 80%, one per condition, HG underneath."""
    conds = [c for c in cfg.COND_SHORT if c in per]
    if not conds:
        return
    w0, w1 = cfg.ersp_display_window
    fig, axes = plt.subplots(2, len(conds), figsize=(2.5 * len(conds), 5.2), dpi=140,
                             squeeze=False, gridspec_kw=dict(height_ratios=[2.4, 1.0]))
    for j, c in enumerate(conds):
        v = per[c]
        ax = axes[0][j]
        ax.pcolormesh(v["x"], v["f"], v["db"], cmap="bwr", vmin=-cfg.ersp_vlim,
                      vmax=cfg.ersp_vlim, shading="auto")
        ax.axvline(0, color="k", lw=0.9)
        ax.set_xlim(w0, w1)
        ax.set_box_aspect(1)
        ax.set_title(f"{c}  (n={v['n']}{', -' + str(v['dropped']) if v['dropped'] else ''})",
                     fontsize=7.5, color=INK)
        if j == 0:
            ax.set_ylabel("Hz", fontsize=7)
        ax = axes[1][j]
        ax.plot(v["x"], v["hg"], lw=1.0, color="#c1121f")
        ax.axhline(0, color=GREY, lw=0.6)
        ax.axvline(0, color="k", lw=0.9)
        ax.set_xlim(w0, w1)
        ax.set_xlabel("s from GO", fontsize=6.5)
        if j == 0:
            ax.set_ylabel("HG (dB)", fontsize=7)
    for a in fig.axes:
        a.tick_params(labelsize=6, colors=MUTED, length=2)
    fig.suptitle(f"{pat}  {ch}   ERSP 0-{cfg.fmax:.0f} Hz (RT, no warping), "
                 f"baseline 20% / GO 80%, dB vs {cfg.baseline_calc_w[0]:+.1f}"
                 f"..{cfg.baseline_calc_w[1]:+.1f} s", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(png, facecolor="white")
    plt.close(fig)


def plot_contrast(pat, summary, png, wm, reref):
    if not summary:
        return
    rows = []
    for ch, v in summary.items():
        foot = np.nanmean([v.get(c, np.nan) for c in ("foot_left", "foot_right")])
        hand = np.nanmean([v.get(c, np.nan) for c in ("hand_left", "hand_right")])
        rows.append((re.sub(r"\d+$", "", ch), ch, v.get("mouth", np.nan) - foot,
                     hand - foot, ch in wm))
    shafts = sorted({r[0] for r in rows})
    fig, ax = plt.subplots(figsize=(11, 0.34 * len(shafts) + 2.4), dpi=150)
    for k, sh in enumerate(shafts):
        gm = [r[2] for r in rows if r[0] == sh and not r[4]]
        wmv = [r[2] for r in rows if r[0] == sh and r[4]]
        ctrl = [r[3] for r in rows if r[0] == sh]
        ax.plot(gm, [k] * len(gm), "o", ms=3.4, color="#c1121f", alpha=0.75)
        ax.plot(wmv, [k] * len(wmv), "o", ms=4.4, mfc="none", mec="#1b7837", mew=1.2)
        ax.plot(ctrl, [k + 0.28] * len(ctrl), "o", ms=2.6, color=GREY, alpha=0.7)
        if gm + wmv:
            ax.plot(np.nanmean(gm + wmv), k, "|", ms=13, color=INK, mew=1.7)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(range(len(shafts)))
    ax.set_yticklabels(shafts, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"mean {cfg.broadband[0]:.0f}-{cfg.broadband[1]:.0f} Hz dB, "
                  f"0.2-{cfg.design_stim_s:.0f} s after GO", fontsize=8, color=MUTED)
    ax.set_title(f"{pat}  reref={reref}   red = Mouth minus Foot   ·   green rings = white "
                 f"matter (control run only)   ·   grey = Hand minus Foot",
                 fontsize=9, color=INK, loc="left")
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    for sp in ax.spines.values():
        sp.set_color(GREY)
    fig.tight_layout()
    fig.savefig(png, facecolor="white")
    plt.close(fig)


def write_wm_review(pat, pid, wm_per, summary, out_dir, per_page=8):
    """Every WM contact as if it were data, eight to a page, plus a table with a flag.

    The flag is a prompt to look, not a verdict: broadband more than 1 dB from zero in any
    condition, or mouth-minus-foot beyond 1 dB. A contact that is genuinely in white matter
    should do neither; one that does is a candidate for config_mm.wm_exclude.
    """
    conds = list(cfg.COND_SHORT)
    rows = []
    for ch in sorted(wm_per, key=lambda s: (re.sub(r"\d+$", "", s), int(re.sub(r"\D", "", s) or 0))):
        v = summary.get(ch, {})
        foot = np.nanmean([v.get(c, np.nan) for c in ("foot_left", "foot_right")])
        md = v.get("mouth", np.nan) - foot
        big = max((abs(v.get(c, 0.0)) for c in conds), default=0.0)
        rows.append((ch, v, md, big, (big > 1.0) or (abs(md) > 1.0)))
    tsv = os.path.join(out_dir, "WM_review.tsv")
    with open(tsv, "w", encoding="utf-8") as fh:
        fh.write("contact\t" + "\t".join(conds) + "\tmouth_minus_foot\tmax_abs\tresponds\n")
        for ch, v, md, big, flag in rows:
            fh.write(ch + "\t" + "\t".join(f"{v.get(c, float('nan')):.2f}" for c in conds)
                     + f"\t{md:.2f}\t{big:.2f}\t{'YES' if flag else ''}\n")
    n_flag = sum(1 for r in rows if r[4])
    print(f"  WM review: {len(rows)} contacts, {n_flag} flagged as responding -> {tsv}")

    w0, w1 = cfg.ersp_display_window
    pages = [rows[i:i + per_page] for i in range(0, len(rows), per_page)]
    for pg, chunk in enumerate(pages, 1):
        fig, axes = plt.subplots(len(chunk), 3, figsize=(9.6, 1.9 * len(chunk) + 0.6),
                                 dpi=130, squeeze=False)
        for r, (ch, v, md, big, flag) in enumerate(chunk):
            per = wm_per[ch]
            for j, c in enumerate(("mouth", "foot_left")):
                ax = axes[r][j]
                d = per.get(c)
                if d is None:
                    ax.axis("off")
                    continue
                ax.pcolormesh(d["x"], d["f"], d["db"], cmap="bwr", vmin=-cfg.ersp_vlim,
                              vmax=cfg.ersp_vlim, shading="auto")
                ax.axvline(0, color="k", lw=0.8)
                ax.set_xlim(w0, w1)
                ax.set_title(f"{ch}  {c}", fontsize=7.5, color=("#c1121f" if flag else INK))
                ax.tick_params(labelsize=5.5, colors=MUTED, length=1.5)
            ax = axes[r][2]
            for c in conds:
                d = per.get(c)
                if d is not None:
                    ax.plot(d["x"], d["hg"], lw=0.9,
                            color={"hand_left": "#4a6fa5", "hand_right": "#7fa1c9",
                                   "foot_left": "#1b7837", "foot_right": "#7bbf8a",
                                   "mouth": "#c1121f"}[c], label=c)
            ax.axhline(0, color=GREY, lw=0.5)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlim(w0, w1)
            ax.set_title(f"HG   mouth-foot {md:+.2f} dB   max |x| {big:.2f}"
                         + ("   <- RESPONDS" if flag else ""),
                         fontsize=7.5, color=("#c1121f" if flag else INK))
            ax.tick_params(labelsize=5.5, colors=MUTED, length=1.5)
            if r == 0:
                ax.legend(fontsize=5, frameon=False, ncol=3)
        fig.suptitle(f"{pat}  white-matter contacts as data (unreferenced)  -  "
                     f"page {pg}/{len(pages)}   red title = flagged", fontsize=9, color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"WM_review_{pg}.png"), facecolor="white")
        plt.close(fig)
    print(f"  wrote {len(pages)} WM review page(s)")


# ---------------------------------------------------------------------------
def run(pat, reref="wm", only=None, contrast_only=False):
    lm_cfg, fe, io = lm()
    pre = cfg.MM_PRESETS[pat]
    pid = pre["pat_name"]
    off = pre.get("align_macro_offset_s")
    if off is None:
        raise SystemExit(f"{pat}: no align_macro_offset_s in config_mm.py - the cue "
                         f"times are on the Blackrock clock and cannot be carried across")
    tr = load_trials(pat, pre)
    path = trc_path(pre)
    meta = read_trc_meta(path)
    print(f"\n=== {pat} ({pid})  macro, LM chain, reref={reref}")
    print(f"  TRC {os.path.basename(path)}  {meta['n_chan']} ch @ {meta['fs']:.0f} Hz, "
          f"{meta['dur']:.1f} s;  {len(tr)} trials, clock offset {off:+.1f} s")

    # cue times on the Micromed clock, in samples
    fs = meta["fs"]
    on_s = tr.onset_s.to_numpy() + off
    if on_s.min() < 2 or on_s.max() > meta["dur"] - 6:
        raise SystemExit(f"mapped onsets span {on_s.min():.1f}..{on_s.max():.1f} s in a "
                         f"{meta['dur']:.1f} s file - the offset looks wrong")
    onsets = np.round(on_s * fs).astype(np.int64)
    offsets = np.round((tr.offset_s.to_numpy() + off) * fs).astype(np.int64)
    tends = np.round((tr.trial_end_s.to_numpy() + off) * fs).astype(np.int64)
    cond = tr.condition.to_numpy()

    print("  reading the TRC (one pass, native rate) ...")
    X, names, fs = load_trc(meta)
    n_samp = X.shape[0]

    # ---- 1. bad channels, exactly as 140 ---------------------------------------
    manual = list((getattr(lm_cfg, "bad_channels_manual", {}) or {}).get(pid, []))
    # MM-only WM exclusions join the same list LM's reference builder already honours
    wm_out = list((getattr(cfg, "wm_exclude", {}) or {}).get(pat, []))
    if wm_out:
        print(f"  {len(wm_out)} WM contacts excluded from the reference by config_mm."
              f"wm_exclude: {', '.join(wm_out)}")
    manual_for_ref = sorted(set(manual) | set(wm_out))
    auto, _info = fe.suggest_bad_channels_by_psd(
        X, fs, names, mains_base=getattr(lm_cfg, "mains_base", 50.0), fmax=cfg.fmax,
        line_z=getattr(lm_cfg, "line_z", 3.0),
        total_power_z=getattr(lm_cfg, "total_power_z", 3.5),
        flat_var_thresh=getattr(lm_cfg, "flat_var_thresh", 1e-10))
    bad = sorted(set(manual) | set(auto))
    print(f"  bad channels: {len(manual)} manual + {len(auto)} auto -> {len(bad)}"
          + (f"  ({', '.join(bad[:12])}{'...' if len(bad) > 12 else ''})" if bad else ""))

    # ---- 2. WM reference, exactly as 140 ---------------------------------------
    lm_pre = (getattr(lm_cfg, "MICROEPI_MAT_PRESETS", {}) or {}).get(pat, {})
    wm_idx = io.wm_indices_for_patient(pid, names,
                                       electrodes_tsv_pattern=lm_pre.get("electrodes_tsv"))
    wm_names = {names[i] for i in wm_idx}
    skip = set(bad)
    if reref == "wm":
        if not wm_idx:
            raise SystemExit(f"{pid}: no WM channels found - cannot WM-reference. "
                             f"Use --reref none for the unreferenced control.")
        X, used, excluded = fe.apply_wm_reference_with_exclusions(
            X, names, wm_idx, manual_for_ref,
            method=getattr(lm_cfg, "wm_ref_method", "mean"),
            min_wm=getattr(lm_cfg, "wm_min_contacts", 3))
        skip |= set(used) | set(excluded)
        print(f"  WM reference from {len(used)} contacts ({len(excluded)} WM excluded as "
              f"bad); the {len(skip)} reference + bad contacts are not analysed")
    else:
        print(f"  no re-referencing: {len(wm_names)} WM contacts stay in as the control")

    # ---- 3. per block: notch + PSD, exactly as 140 -------------------------------
    conds = list(cfg.COND_SHORT)
    out_dir = os.path.join(ROOT, "outputs", pat, "macro", reref)
    os.makedirs(out_dir, exist_ok=True)
    pad = float(getattr(lm_cfg, "notch_block_pad_s", 10.0))
    per_shaft = pid in getattr(lm_cfg, "notch_shaft_patients", [])
    getd = lambda nm, d: ((getattr(lm_cfg, nm, None) or {}).get(pat, d)
                          if isinstance(getattr(lm_cfg, nm, None), dict)
                          else getattr(lm_cfg, nm, d))
    blocks = {}
    for c in conds:
        m = cond == c
        if m.sum() < 3:
            continue
        b0, b1 = fe.block_window(onsets[m], tends[m], n_samp, fs, pad_s=pad)
        sig = np.ascontiguousarray(X[b0:b1])
        fe.plot_psd_overview(sig, fs, names,
                             save_root=os.path.join(out_dir, "PSD_raw", c),
                             patient_id=pid, block_name=f"MM {c} (raw)",
                             fmax=cfg.fmax, mains_base=getattr(lm_cfg, "mains_base", 50.0),
                             dpi=cfg.psd_dpi)
        audit = []
        sig = fe.apply_notch_with_audit(
            sig, fs, pid, pat,
            notch_patients=getattr(lm_cfg, "notch_patients", ()),
            mains_base=getattr(lm_cfg, "mains_base", 50.0), fmax=cfg.fmax,
            repeats=getattr(lm_cfg, "notch_repeats", 1),
            peak_z_thresh=getattr(lm_cfg, "notch_peak_z_thresh", 3.0),
            extra_bases=tuple(getd("notch_extra_bases", ()) or ()),
            audit=audit, per_shaft=per_shaft, names=names,
            Q_max=float(getd("notch_Q_max", 500.0)),
            method=str(getd("notch_method", "iir")),
            interp_kw=dict(phase=getattr(lm_cfg, "notch_interp_phase", "random"),
                           max_hw_hz=getattr(lm_cfg, "notch_interp_max_hw_hz", 12.0)))
        fe.plot_psd_overview(sig, fs, names,
                             save_root=os.path.join(out_dir, "PSD_clean", c),
                             patient_id=pid, block_name=f"MM {c} (after notch)",
                             fmax=cfg.fmax, mains_base=getattr(lm_cfg, "mains_base", 50.0),
                             dpi=cfg.psd_dpi)
        if audit:
            ap = os.path.join(out_dir, f"{pid}_{c}_notch_audit.tsv")
            keys = sorted({k for r in audit for k in r})
            with open(ap, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
                w.writeheader()
                w.writerows(audit)
        blocks[c] = dict(sig=sig, on=onsets[m] - b0, off=offsets[m] - b0, te=tends[m] - b0,
                         b0=b0, n_notch=len(audit))
        print(f"  {c:11s} block {b0/fs:7.1f}..{b1/fs:7.1f} s, {int(m.sum())} trials, "
              f"{len(audit)} harmonics notched")

    # ---- 4. per channel: compute_ersp (RT) + plot_hg_trials, exactly as 140 ------
    params = fe.ERSPParams(nperseg=cfg.nperseg, nfft=cfg.nfft, fmax=cfg.fmax,
                           baseline_w=cfg.baseline_w, baseline_calc_w=cfg.baseline_calc_w)
    params.trial_reject_mad = (cfg.ersp_trial_reject or {}).get(pat)
    params.trial_reject_z = (cfg.ersp_trial_reject_z or {}).get(pat)
    params.trial_reject_hg_mad = (cfg.ersp_trial_reject_hg_mad or {}).get(pat)
    params.trial_reject_hg_z = (cfg.ersp_trial_reject_hg_z or {}).get(pat)

    want = None
    if only:
        want = {w.strip().lower() for w in only.split(",")}
    summary = {}
    wm_per = {}                      # the WM contacts' full results, for the review sheet
    hg_dir = os.path.join(out_dir, "HG")
    ersp_dir = os.path.join(out_dir, "ERSP")
    for d in (hg_dir, ersp_dir):
        os.makedirs(d, exist_ok=True)
    for ci, ch in enumerate(names):
        if ch in skip or (want and ch.lower() not in want):
            continue
        per = {}
        for c, blk in blocks.items():
            res = fe.compute_ersp(blk["sig"], fs, blk["on"], blk["off"], ci,
                                  trial_ends=blk["te"], mode="RT",
                                  time_window=cfg.time_window, params=params)
            f, x, db = res["f"], np.asarray(res["x"], float), res["avg_db"]
            band = (f >= cfg.hg_band[0]) & (f <= cfg.hg_band[1])
            per[c] = dict(f=f, x=x, db=db, hg=np.nanmean(db[band], axis=0),
                          n=int(res.get("n_trials_used", res["meta"].get("n_trials", 0))),
                          dropped=int(res.get("n_dropped", 0)))
            if not contrast_only:
                fe.plot_hg_trials(blk["sig"], fs, blk["on"], blk["off"], ci,
                                  chan_name=ch, patient_id=pid, condition=c,
                                  reref_type=("WM" if reref == "wm" else "none"),
                                  time_window=cfg.time_window, baseline_w=cfg.baseline_w,
                                  hg_band=cfg.hg_band, smooth_ms=cfg.hg_smooth_ms,
                                  rejected_trials=res.get("dropped_trials"),
                                  vmin=cfg.hg_vmin, vmax=cfg.hg_vmax,
                                  save_dir=os.path.join(hg_dir, c),
                                  trial_end_indices=blk["te"], sort_by="stim")
        if not per:
            continue
        f = per[next(iter(per))]["f"]
        bb = (f >= cfg.broadband[0]) & (f <= cfg.broadband[1])
        summary[ch] = {}
        for c, v in per.items():
            resp = (v["x"] >= 0.2) & (v["x"] <= cfg.design_stim_s)
            summary[ch][c] = float(np.nanmean(v["db"][np.ix_(bb, resp)]))
        if ch in wm_names:
            wm_per[ch] = per
        if not contrast_only:
            plot_ersp_square(pat, ch, per, os.path.join(ersp_dir, f"{ch}.png"))

    # ---- summary -----------------------------------------------------------------
    p = os.path.join(out_dir, f"{pid}_MM_broadband.tsv")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("channel\ttissue\t" + "\t".join(conds) + "\tmouth_minus_foot\n")
        for ch, v in summary.items():
            foot = np.nanmean([v.get(c, np.nan) for c in ("foot_left", "foot_right")])
            fh.write(f"{ch}\t{'wm' if ch in wm_names else 'gm'}\t"
                     + "\t".join(f"{v.get(c, float('nan')):.3f}" for c in conds)
                     + f"\t{v.get('mouth', np.nan) - foot:.3f}\n")
    plot_contrast(pat, summary, os.path.join(out_dir, f"{pid}_MM_contrast.png"),
                  wm_names, reref)
    if reref == "none" and wm_per:
        write_wm_review(pat, pid, wm_per, summary, out_dir)
    md = {ch: v.get("mouth", np.nan) -
          np.nanmean([v.get(c, np.nan) for c in ("foot_left", "foot_right")])
          for ch, v in summary.items()}
    a_wm = np.array([d for ch, d in md.items() if ch in wm_names], float)
    a_gm = np.array([d for ch, d in md.items() if ch not in wm_names], float)
    print(f"\n  mouth minus foot, {cfg.broadband[0]:.0f}-{cfg.broadband[1]:.0f} Hz "
          f"({len(summary)} contacts analysed):")
    if len(a_gm):
        print(f"    grey matter  ({len(a_gm):3d}): mean {np.nanmean(a_gm):+.2f}  "
              f"median {np.nanmedian(a_gm):+.2f}  {int((a_gm > 0.5).sum())} above +0.5")
    if len(a_wm):
        print(f"    white matter ({len(a_wm):3d}): mean {np.nanmean(a_wm):+.2f}  "
              f"median {np.nanmedian(a_wm):+.2f}  {int((a_wm > 0.5).sum())} above +0.5"
              f"   <- the control; should be ~0")
    print(f"  wrote {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="G-05")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--reref", default="wm", choices=["wm", "none"],
                    help="wm = the LM chain (default); none = the white-matter control")
    ap.add_argument("--channels", default=None)
    ap.add_argument("--contrast-only", action="store_true")
    a = ap.parse_args()
    for p in (cfg.patient_ids if a.all else [a.patient]):
        try:
            run(p, reref=a.reref, only=a.channels, contrast_only=a.contrast_only)
        except SystemExit as e:
            print(f"  !! {p}: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  !! {p} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
