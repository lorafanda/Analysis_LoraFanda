"""Step 1c - the timing, seen: micro and macro contacts on one clock, with the trigger
channel each system recorded, one window per part.

    python .\\12_mm_timing_check.py --patient G-05
    python .\\12_mm_timing_check.py --patient G-05 --shafts AG HPG --win 25

Needs outputs/<pat>/sync_parts.tsv (11_mm_sync.py), the photodiode cache (10_mm_triggers.py)
and the micro cache (21_mm_micro.py, any run). Reads the TRC directly.

WHAT IS DRAWN. One column per fitted part, one window of --win s starting 3 s before a
cue in the middle of that part, everything on the TRC clock:
  1. the Blackrock photodiode (ainp1), carried over with that part's offset; the trial
     table's stimulus intervals shaded. This is the cue.
  2. the Micromed MKR4+ trace, with the .nev sync bursts (carried over) as ticks. Both
     systems' sync events: they should sit on top of each other.
  3+. per shaft: the micro contacts' mean and the first three macro contacts' mean, both
     1-40 Hz and z-scored, overlaid. Same tissue a millimetre apart - the slow waves must
     coincide. The macro with the OLD single offset is drawn dashed grey in the first
     shaft panel so the gap is visible, and the residual lag of macro vs micro from the
     cross-correlation of the whole part is in the panel title.

If the three agree - bursts on pulses, slow waves on slow waves, cue where the log says -
the per-part offsets are right and the macro can be cut.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly, correlate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "functions"))
import config_mm as cfg                                                      # noqa: E402
from lf_mm_io import (read_trc_meta, pull_channels, load_cache, session_timeline,  # noqa: E402
                      part_paths, real_parts, part_num, part_index, read_nev_digital)

INK, MUTED, GREY, RED, BLUE = "#1b232c", "#68727d", "#c9ced4", "#c1121f", "#2b5f9e"


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


def bp(x, fs, lo=1.0, hi=40.0):
    b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def z(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def lag_s(u, v, fs, max_s=8.0):
    """Lag of v relative to u (s) at the peak |cross-correlation| within +-max_s."""
    n = min(len(u), len(v))
    uu, vv = u[:n] - u[:n].mean(), v[:n] - v[:n].mean()
    c = correlate(uu, vv, mode="full", method="fft")
    lags = (np.arange(len(c)) - (n - 1)) / fs
    w = np.abs(lags) < max_s
    return float(lags[w][np.argmax(np.abs(c[w]))])


def run(pat, shafts=None, win=25.0):
    import pandas as pd
    pre = cfg.MM_PRESETS[pat]
    pid = pre["pat_name"]
    out_dir = os.path.join(ROOT, "outputs", pat)
    sp = os.path.join(out_dir, "sync_parts.tsv")
    if not os.path.exists(sp):
        raise SystemExit(f"no {sp}\n  run:  python .\\11_mm_sync.py --patient {pat}")
    sync = pd.read_csv(sp, sep="\t", dtype={"part": str})
    by = {str(r.part): float(r.offset_s) for r in sync.itertuples()}
    off_old = float(pre["align_macro_offset_s"])
    tr = pd.read_csv(os.path.join(out_dir, f"{pid}_MM_trials.tsv"), sep="\t")

    parts, _tot = session_timeline(part_paths(pre))
    rp = real_parts(parts)
    names_rp = [part_num(p) for p in rp]
    pi = part_index(tr.onset_s.to_numpy(), parts)

    pd_arr, pd_meta = load_cache(os.path.join(out_dir, "photodiode.npy"))
    if pd_arr is None:
        raise SystemExit("no photodiode cache - run 10_mm_triggers.py first")
    pd_fs = float(pd_meta["fs"])
    mi_arr, mi_meta = load_cache(os.path.join(out_dir, f"micro_lfp_{int(cfg.micro_lfp_fs)}.npy"))
    if mi_arr is None:
        raise SystemExit("no micro cache - run 21_mm_micro.py first")
    mi_fs, mi_names, mi_t0 = float(mi_meta["fs"]), list(mi_meta["names"]), float(mi_meta["t0"])

    meta = read_trc_meta(trc_path(pre))
    fs_m = meta["fs"]
    micro_shafts = sorted({re.match(r"^(.*?)\d+$", n).group(1) for n in mi_names
                           if re.match(r"^(.*?)\d+$", n)})
    if shafts is None:
        shafts = [s[:-1] for s in micro_shafts if s.endswith("m")]
    pairs = []
    for ma in shafts:
        mi = ma + "m"
        ui = [i for i, n in enumerate(mi_names) if n.startswith(mi)]
        ci = [i for i, n in enumerate(meta["names"]) if re.match(rf"^{ma}\d+$", n)][:3]
        if ui and ci:
            pairs.append((ma, ui, ci))
    if not pairs:
        raise SystemExit(f"no micro/macro shaft pairs among {shafts}")
    sync_ch = [i for i, n in enumerate(meta["names"]) if n.upper().startswith("MKR4")]
    idxs = sorted({i for _m, _u, ci in pairs for i in ci} | set(sync_ch))
    print(f"  reading {len(idxs)} TRC channels ...")
    got, _ = pull_channels(meta, idxs, target_fs=None, dtype=meta["dtype"])

    # bursts per part, part clock
    bursts = {}
    for p in rp:
        nv = p["path"][:-4] + ".nev"
        if part_num(p) in by and os.path.exists(nv):
            t = read_nev_digital(nv)["t"]
            bursts[part_num(p)] = t[np.r_[0, np.flatnonzero(np.diff(t) > 1.0) + 1]] if len(t) else t

    cols = [k for k, p in enumerate(rp) if names_rp[k] in by and (pi == k).any()]
    if not cols:
        raise SystemExit("no fitted part holds trials")
    nrow = 2 + len(pairs)
    fig, axes = plt.subplots(nrow, len(cols), figsize=(6.4 * len(cols), 1.75 * nrow + 0.8),
                             dpi=130, squeeze=False)
    FS = 200.0
    for c, k in enumerate(cols):
        p, name = rp[k], names_rp[k]
        off = by[name]
        sel = np.flatnonzero(pi == k)
        t_cue = float(tr.onset_s.iloc[sel[len(sel) // 2]])            # a cue mid-part
        a0, a1 = t_cue - 3.0, t_cue - 3.0 + win                        # concat clock
        a0 = max(a0, p["t0"] + 0.5, mi_t0 + 0.5)
        a1 = min(a1, p["t0"] + p["dur"] - 0.5, mi_t0 + mi_arr.shape[1] / mi_fs - 0.5)
        b0, b1 = a0 + off, a1 + off                                     # TRC clock
        stim = tr[(tr.onset_s >= a0) & (tr.onset_s < a1)]

        # 1. photodiode
        ax = axes[0][c]
        s = slice(int(a0 * pd_fs), int(a1 * pd_fs))
        tt = np.arange(s.start, s.stop) / pd_fs + off
        ax.plot(tt, pd_arr[s], lw=0.6, color=INK)
        for on, of in zip(stim.onset_s, stim.offset_s):
            ax.axvspan(on + off, of + off, color=RED, alpha=0.13, lw=0)
        ax.set_title(f"part {name}   offset {off:+.3f} s (old single offset {off_old:+.1f})   "
                     f"session {a0:.0f}-{a1:.0f} s -> TRC {b0:.0f}-{b1:.0f} s",
                     fontsize=8.5, color=INK, loc="left")
        ax.set_ylabel("photodiode\n(Blackrock)", fontsize=7.5, color=MUTED)

        # 2. MKR4+ and the .nev bursts
        ax = axes[1][c]
        if sync_ch:
            x = got[sync_ch[0]][int(b0 * fs_m):int(b1 * fs_m)].astype(float)
            x -= np.median(x)
            ax.plot(np.arange(len(x)) / fs_m + b0, x, lw=0.5, color=INK)
        bt = bursts.get(name, np.array([])) + p["t0"] + off               # part -> concat -> TRC
        for t in bt[(bt > b0) & (bt < b1)]:
            ax.axvline(t, color=RED, lw=1.2, ls=(0, (3, 2)))
        ax.set_ylabel("MKR4+ (Micromed)\n+ .nev bursts (red)", fontsize=7.5, color=MUTED)

        # 3+. shafts
        for r, (ma, ui, ci) in enumerate(pairs):
            ax = axes[2 + r][c]
            ms = slice(int((a0 - mi_t0) * mi_fs), int((a1 - mi_t0) * mi_fs))
            u = bp(mi_arr[ui].mean(0)[ms].astype(float), mi_fs)
            u = resample_poly(u, 1, int(mi_fs / FS))
            v = np.mean([got[i][int(b0 * fs_m):int(b1 * fs_m)].astype(float) for i in ci], axis=0)
            v = bp(v - v.mean(), fs_m)
            v = resample_poly(v, int(FS), int(fs_m))
            n = min(len(u), len(v))
            tt = np.arange(n) / FS + b0
            ax.plot(tt, z(u[:n]), lw=0.7, color=BLUE, label=f"{ma}m1-{len(ui)} mean (micro)")
            ax.plot(tt, z(v[:n]), lw=0.7, color=RED, alpha=0.85,
                    label=f"{ma}{'/'.join(meta['names'][i][len(ma):] for i in ci)} mean (macro)")
            if r == 0:
                w = np.mean([got[i][int((a0 + off_old) * fs_m):int((a1 + off_old) * fs_m)]
                             .astype(float) for i in ci], axis=0)
                w = resample_poly(bp(w - w.mean(), fs_m), int(FS), int(fs_m))
                ax.plot(tt, z(w[:n]) - 4.5, lw=0.6, color=GREY, ls="--",
                        label="macro with the OLD single offset (shifted down)")
            # whole-part residual lag
            pa0, pa1 = max(p["t0"] + 1, mi_t0 + 1), min(p["t0"] + p["dur"] - 1,
                                                          mi_t0 + mi_arr.shape[1] / mi_fs - 1)
            if pa1 - pa0 > 60:
                s2 = slice(int((pa0 - mi_t0) * mi_fs), int((pa1 - mi_t0) * mi_fs))
                U = resample_poly(bp(mi_arr[ui].mean(0)[s2].astype(float), mi_fs), 1, int(mi_fs / FS))
                V = np.mean([got[i][int((pa0 + off) * fs_m):int((pa1 + off) * fs_m)].astype(float)
                             for i in ci], axis=0)
                V = resample_poly(bp(V - V.mean(), fs_m), int(FS), int(fs_m))
                lag = lag_s(U, V, FS)
                ax.set_title(f"{ma}: macro lags micro by {lag:+.2f} s over the whole part "
                             f"(0 = aligned)", fontsize=8, color=INK, loc="left")
            for on in stim.onset_s:
                ax.axvline(on + off, color=GREY, lw=0.6)
            ax.set_ylabel(f"{ma} / {ma}m\n1-40 Hz, z", fontsize=7.5, color=MUTED)
            if c == 0 and r == 0:
                handles, labels = ax.get_legend_handles_labels()
        for r in range(nrow):
            axes[r][c].set_xlim(b0, b1)
            axes[r][c].tick_params(labelsize=6.5, colors=MUTED, length=2)
            for spn in axes[r][c].spines.values():
                spn.set_color(GREY)
        axes[-1][c].set_xlabel("TRC clock (s)", fontsize=8, color=MUTED)

    fig.suptitle(f"{pat}   timing check: Blackrock (photodiode, .nev bursts, micro LFP) carried "
                 f"to the Micromed clock with each part's own offset, against MKR4+ and the macro "
                 f"LFP", fontsize=10, color=INK)
    fig.legend(handles, ["micro: shaft mean", "macro: first three contacts, mean",
                         "macro with the OLD single offset (drawn 4.5 below)"],
               loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    png = os.path.join(out_dir, "timing_check.png")
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    print(f"  wrote {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="G-05")
    ap.add_argument("--shafts", nargs="*", default=None,
                    help="macro shaft names that have micro wires (default: all)")
    ap.add_argument("--win", type=float, default=25.0, help="window length per part, s")
    a = ap.parse_args()
    run(a.patient, shafts=a.shafts, win=a.win)


if __name__ == "__main__":
    main()
