"""Step 3 - the Blackrock micro analysis: referenced, notched, and plotted by tetrode.

    python .\\21_mm_micro.py --patient G-05 --reref shaft_mean
    python .\\21_mm_micro.py --patient G-05 --reref first
    python .\\21_mm_micro.py --patient G-05 --reref none --no-notch     (what it looks like raw)
    python .\\21_mm_micro.py --patient G-05 --reref shaft_mean --ersp   (per-condition ERSPs too)
    python .\\21_mm_micro.py --patient G-05 --force                     (re-read the .ns6 parts)

WHY THIS EXISTS IN THIS FORM. The first pass at the micro ERSPs was unreadable: horizontal
bands of mains harmonics across the whole range, every panel uniformly blue, no visible
task response. That is a picture of the noise floor. Three things were wrong and they
interact, so all three are handled here.

  REFERENCING. The micro contacts are unreferenced as recorded, so a shared line or a
  drifting ground sits on all four wires of a tetrode at once and survives averaging.
  Two schemes are available and they fail differently, which is why both are runnable:
    --reref first        subtract the first contact of the shaft (AGm1 from all AGm).
                         Keeps one intact reference, but hands that contact's own noise
                         to every other contact with the sign flipped. The reference
                         becomes identically zero and is dropped from the figures rather
                         than drawn as a flat line pretending to be data.
    --reref shaft_mean   subtract the mean of the shaft. Spreads that risk over all
                         contacts, but also subtracts whatever the shaft genuinely
                         shares - which for four wires tens of microns apart may be a
                         real local field, not only noise.
  Running both and comparing is the check; a response that survives either is not a
  referencing artefact.

  NOTCH. The same adaptive notch the LM pipeline uses, imported from
  01_FBM_Analysis/functions/lf_ersp.py rather than reimplemented, with its settings read
  from the LM config: the mains comb is a property of the room and the amplifier, not of
  the task. It is applied PER SHAFT, and it only removes a harmonic whose spectral peak
  clears the z threshold, so a clean harmonic is left alone.

  BANDWIDTH. Clean further up than you look. Harmonics above the display limit still fold
  energy into the estimate through the STFT window, so the notch runs to 2000 Hz while the
  ERSP is drawn to 1000. Both need headroom the old 1000 Hz cache did not have, so the
  cache is now 5000 Hz - and, because that is five times the data, windowed to the task.

NO CLOCK ALIGNMENT IS INVOLVED. The cue times come from the photodiode on ainp1 and the
micro signal from the same Blackrock recording: one clock, one origin. The macro analysis
has to carry cue times across a measured offset good to about +-0.35 s; this one does not,
so its timing is exact and it is the better place to compare response latencies.

ANALOG INPUTS. The Blackrock ainp channels ride along (config micro_include_ainp): ainp1
is the photodiode, ainp2/3 are whatever was plugged in - in G-05 nothing, they are
identical to each other and white. They go through the same cache, notch, epoching, HG
and ERSP, but they are not on a shaft: never a reference, never re-referenced. Their
figure is the "analog" group. The photodiode's ERSP is the cue itself, a timing check;
the unconnected inputs are the amplifier with no tissue attached, the Blackrock
counterpart of X1/X5 on the Micromed.

TETRODE LAYOUT. Contacts come in fours and four wires tens of microns apart see
overlapping but not identical populations. Plotting them 2x2 keeps that in front of you:
one corner moving alone is local, all four moving together is a shared field or something
instrumental. A flat list of 48 panels hides exactly that distinction.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
from scipy.signal import spectrogram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "functions"))
# Stage 01 on the path so the notch and its settings come from the LM code, not a copy.
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "..", "01_FBM_Analysis")))
import config_mm as cfg                                                  # noqa: E402
from lf_mm_io import (read_nsx_meta, session_timeline, pull_channels,    # noqa: E402
                      save_cache, load_cache, crosses_splice)

INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"
COND_COL = {"hand_left": "#4a6fa5", "hand_right": "#7fa1c9",
            "foot_left": "#1b7837", "foot_right": "#7bbf8a",
            "mouth": "#c1121f"}


# ---------------------------------------------------------------------------
# raw -> cached LFP
# ---------------------------------------------------------------------------
def part_paths(pre):
    folder = pre["blackrock_dir"]
    first, last = pre["blackrock"]
    stem, a = first.rsplit("-", 1)
    b = last.rsplit("-", 1)[1]
    return [os.path.join(folder, f"{stem}-{n}.ns6") for n in range(int(a), int(b) + 1)]


def is_ainp(name):
    n = str(name).lower()
    return n.startswith("ainp") or n == "photodiode"


def micro_names(meta):
    return [n for n in meta["names"] if n and not is_ainp(n)]


def ainp_names(meta):
    return [n for n in meta["names"] if n and is_ainp(n)]


def cache_names(meta):
    """Micro contacts first, then the analog inputs if the config asks for them."""
    names = micro_names(meta)
    if getattr(cfg, "micro_include_ainp", False):
        names += ainp_names(meta)
    return names


def extract_lfp(pat, pre, force=False):
    """Micro channels over the task window, decimated to cfg.micro_lfp_fs, cached.

    Only the task window is kept. At 5000 Hz the whole session is about 1.5 GB per
    patient and nothing outside the window is ever epoched, so the cache carries the
    window plus a margin and records where it starts on the session clock.
    """
    fs_t = float(cfg.micro_lfp_fs)
    cache = os.path.join(ROOT, "outputs", pat, f"micro_lfp_{int(fs_t)}.npy")
    parts, total = session_timeline(part_paths(pre))
    names = cache_names(read_nsx_meta(parts[0]["path"])) if parts else None
    if not force:
        arr, meta = load_cache(cache)
        if arr is not None:
            if names is None:
                print(f"  raw parts not reachable under {pre['blackrock_dir']}; using the "
                      f"cache as it is: {arr.shape} @ {meta['fs']:.0f} Hz")
                return arr, meta
            if list(meta["names"]) == names:
                print(f"  cached micro LFP: {arr.shape} @ {meta['fs']:.0f} Hz, "
                      f"session t0 {meta['t0']:.1f} s   [--force to re-read]")
                return arr, meta
            # a cache written before the analog inputs were included, or after a config
            # change: the channel set is the cache's identity, so it is re-read
            print(f"  cached micro LFP has {len(meta['names'])} channels, the config now "
                  f"asks for {len(names)} - re-reading the .ns6 parts")

    if not parts:
        raise SystemExit(f"{pat}: no .ns6 parts under {pre['blackrock_dir']}")
    tr = pre.get("time_range")
    if not tr:
        raise SystemExit(f"{pat}: time_range is not set in config_mm.py; the micro cache "
                         f"is windowed to the task and needs it")
    pad = float(cfg.micro_cache_pad_s)
    w0, w1 = float(tr[0]) - pad, float(tr[1]) + pad

    n_ain = sum(is_ainp(n) for n in names)
    print(f"  {len(names) - n_ain} micro channels + {n_ain} analog inputs; caching "
          f"{w0:.0f}..{w1:.0f} s of {total:.0f} s at {fs_t:.0f} Hz")

    chunks, got_t0 = [], None
    for k, p in enumerate(parts, 1):
        pt0, pt1 = p["t0"], p["t0"] + p["dur"]
        if pt1 <= w0 or pt0 >= w1:
            continue                                    # this part is outside the task
        m = read_nsx_meta(p["path"])
        idx = [m["names"].index(n) for n in names if n in m["names"]]
        if len(idx) != len(names):
            print(f"    !! {p['name']}: channel set differs, skipped")
            continue
        got, fs_out = pull_channels(m, idx, target_fs=fs_t)
        block = np.stack([got[i] for i in idx])
        # trim this part to the requested window
        a = max(0, int(round((w0 - pt0) * fs_out)))
        b = min(block.shape[1], int(round((w1 - pt0) * fs_out)))
        if b <= a:
            continue
        if got_t0 is None:
            got_t0 = pt0 + a / fs_out
        chunks.append(block[:, a:b])
        print(f"    [{k}/{len(parts)}] {p['name']}  kept {(b - a) / fs_out:7.1f} s")
    if not chunks:
        raise SystemExit(f"{pat}: no data in {w0:.0f}..{w1:.0f} s")
    arr = np.concatenate(chunks, axis=1).astype(np.float32)
    meta = dict(fs=float(fs_t), names=names, t0=float(got_t0),
                # the part layout the cache was cut from, so the seams are known even when
                # the raw folder is not reachable
                parts=[dict(name=p["name"], t0=float(p["t0"]), dur=float(p["dur"]),
                            origin=p["origin"]) for p in parts])
    save_cache(cache, arr, meta)
    print(f"  cached -> {cache}  {arr.shape}  ({arr.nbytes / 1e9:.2f} GB)")
    return arr, meta


# ---------------------------------------------------------------------------
# referencing and notch
# ---------------------------------------------------------------------------
def shaft_of(name):
    m = re.match(r"^(.*?)\d+$", name)
    return m.group(1) if m else name


def rereference(X, names, scheme):
    """Returns (X2, names2, dropped). Channels are rows."""
    if scheme == "none":
        return X, list(names), []
    groups = {}
    for i, n in enumerate(names):
        groups.setdefault(shaft_of(n), []).append(i)
    Y = X.copy()
    dropped = []
    for sh, idx in groups.items():
        if len(idx) < 2 or is_ainp(names[idx[0]]):
            continue                 # analog inputs: not a shaft, kept as recorded
        if scheme == "first":
            ref = X[idx[0]]
            dropped.append(names[idx[0]])
        elif scheme == "shaft_mean":
            ref = X[idx].mean(axis=0)
        else:
            raise ValueError(f"unknown reref scheme {scheme!r}")
        for i in idx:
            Y[i] = X[i] - ref
    keep = [i for i, n in enumerate(names) if n not in dropped]
    return Y[keep], [names[i] for i in keep], dropped


def apply_lm_notch(X, fs, names, pat, pre):
    """The LM adaptive notch, with the LM config's own settings for this patient.

    Imported lazily: functions.lf_ersp pulls in the rest of stage 01, and if that is not
    importable the analysis should still run un-notched with a clear message rather than
    dying.
    """
    try:
        from functions import config as lm_cfg
        from functions.lf_ersp import apply_notch_with_audit
    except Exception as e:
        print(f"  notch unavailable ({type(e).__name__}: {e}) - running WITHOUT it")
        return X, []
    key, pid = pat, pre["pat_name"]
    getd = lambda name, dflt: (getattr(lm_cfg, name, None) or {}).get(key, dflt) \
        if isinstance(getattr(lm_cfg, name, None), dict) else getattr(lm_cfg, name, dflt)
    audit = []
    sig = np.ascontiguousarray(X.T.astype(np.float64))          # (n_samples, n_channels)
    Y = apply_notch_with_audit(
        sig, fs, pid, key,
        notch_patients=getattr(lm_cfg, "notch_patients", ()),
        mains_base=getattr(lm_cfg, "mains_base", 50.0),
        fmax=float(cfg.micro_notch_fmax),
        repeats=getattr(lm_cfg, "notch_repeats", 1),
        peak_z_thresh=getattr(lm_cfg, "notch_peak_z_thresh", 3.0),
        extra_bases=tuple(getd("notch_extra_bases", ()) or ()),
        audit=audit, per_shaft=True, names=list(names),
        Q_max=float(getd("notch_Q_max", 500.0)),
        method=str(getd("notch_method", "iir")),
        interp_kw=dict(phase=getattr(lm_cfg, "notch_interp_phase", "random"),
                       max_hw_hz=getattr(lm_cfg, "notch_interp_max_hw_hz", 12.0)),
    )
    return np.ascontiguousarray(np.asarray(Y).T.astype(np.float32)), audit


def psd_figure(X, fs, names, out_root, pat, tag):
    """PSD before and after cleaning, via the LM plotter so both stages look the same."""
    try:
        from functions.lf_ersp import plot_psd_overview
    except Exception as e:
        print(f"  PSD plot unavailable ({type(e).__name__}: {e})")
        return
    plot_psd_overview(np.ascontiguousarray(X.T.astype(float)), fs, list(names),
                      save_root=os.path.join(out_root, tag), patient_id=pat,
                      block_name=f"MM micro {tag}", fmax=float(cfg.micro_psd_fmax),
                      mains_base=50.0, dpi=200)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def tetrodes(names):
    """[(label, [up to 4 names])]: the micro contacts by shaft in fours, then the analog
    inputs as their own group ("analog"; the photodiode is named without digits in G-02,
    so they are grouped by kind, not by number)."""
    by, ain = {}, []
    for n in names:
        if is_ainp(n):
            ain.append(n)
            continue
        m = re.match(r"^(.*?)(\d+)$", n)
        if m:
            by.setdefault(m.group(1), []).append((int(m.group(2)), n))
    out = []
    for shaft in sorted(by):
        seq = [n for _k, n in sorted(by[shaft])]
        for i in range(0, len(seq), 4):
            grp = seq[i:i + 4]
            if grp:
                out.append((f"{grp[0]}-{grp[-1][len(shaft):]}", grp))
    for i in range(0, len(ain), 4):
        out.append(("analog" if i == 0 else f"analog_{i // 4 + 1}", ain[i:i + 4]))
    return out


def chan_title(ch, pre):
    """Panel title: the analog inputs say what they are."""
    if ch == pre.get("bk_pd_channel"):
        return ch if "photodiode" in ch.lower() else f"{ch}  (photodiode)"
    if is_ainp(ch):
        return f"{ch}  (analog input)"
    return ch


def epoch(x, fs, onsets, win):
    a, b = int(win[0] * fs), int(win[1] * fs)
    out = []
    for t0 in onsets:
        i = int(round(t0 * fs))
        if i + a < 0 or i + b > len(x):
            continue
        out.append(x[i + a:i + b])
    return np.stack(out) if out else np.zeros((0, b - a))


def hg_z(ep, fs):
    from scipy.signal import butter, filtfilt, hilbert
    b, a = butter(4, [cfg.hg_band[0] / (fs / 2), min(cfg.hg_band[1] / (fs / 2), 0.99)],
                  btype="band")
    env = np.abs(hilbert(filtfilt(b, a, ep, axis=-1), axis=-1))
    k = max(1, int(cfg.hg_smooth_ms / 1000.0 * fs)) | 1
    env = np.apply_along_axis(lambda v: np.convolve(v, np.ones(k) / k, "same"), -1, env)
    t = np.arange(ep.shape[1]) / fs + cfg.time_window[0]
    m = (t >= cfg.baseline_calc_w[0]) & (t < cfg.baseline_calc_w[1])
    mu = env[:, m].mean(axis=1, keepdims=True)
    sd = env[:, m].std(axis=1, keepdims=True)
    return t, (env - mu) / np.where(sd > 0, sd, 1.0)


def ersp_db(ep, fs):
    f, t, S = spectrogram(ep, fs=fs, nperseg=cfg.nperseg, noverlap=cfg.noverlap,
                          nfft=cfg.nfft, axis=-1, mode="psd")
    keep = f <= float(cfg.micro_ersp_fmax)
    f, S = f[keep], S[:, keep, :]
    t = t + cfg.time_window[0]
    base = (t >= cfg.baseline_calc_w[0]) & (t < cfg.baseline_calc_w[1])
    if not base.any():
        base = t < 0
    mu = np.nanmean(S[:, :, base], axis=2, keepdims=True)
    return f, t, np.nanmean(10 * np.log10(np.maximum(S, 1e-20) /
                                          np.maximum(mu, 1e-20)), axis=0)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def plot_tetrode_hg(pat, label, grp, per, png, subtitle, titles=None):
    titles = titles or {}
    w0, w1 = cfg.ersp_display_window
    # the analog panels do not share y: the photodiode's z is hundreds and would flatten
    # the unconnected inputs to a line
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8), dpi=150, sharex=True,
                             sharey=not label.startswith("analog"))
    for ax, ch in zip(axes.ravel(), list(grp) + [None] * 4):
        if ch is None or ch not in per:
            ax.axis("off")
            continue
        for c in cfg.COND_SHORT:
            v = per[ch].get(c)
            if v is not None:
                ax.plot(v["t"], v["hg"], lw=1.0, color=COND_COL[c], label=c)
        ax.axhline(0, color=GREY, lw=0.6)
        ax.axvline(0, color="k", lw=1.0)
        ax.set_xlim(w0, w1)
        ax.set_title(titles.get(ch, ch), fontsize=9, color=INK)
        ax.tick_params(labelsize=7, colors=MUTED, length=2)
        for sp in ax.spines.values():
            sp.set_color(GREY)
    axes[0][0].legend(fontsize=6.2, frameon=False, ncol=2)
    for ax in axes[1]:
        ax.set_xlabel("s from GO   (0 at 20% of the axis)", fontsize=7.5, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("HG (z)", fontsize=8, color=MUTED)
    kind = "analog inputs" if label.startswith("analog") else f"tetrode {label}"
    fig.suptitle(f"{pat}   {kind}   high gamma "
                 f"{cfg.hg_band[0]:.0f}-{cfg.hg_band[1]:.0f} Hz   ·   {subtitle}",
                 fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(png, facecolor="white")
    plt.close(fig)


def plot_tetrode_ersp(pat, label, grp, per, cond, png, subtitle, titles=None):
    titles = titles or {}
    w0, w1 = cfg.ersp_display_window
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.4), dpi=150)
    im = None
    for ax, ch in zip(axes.ravel(), list(grp) + [None] * 4):
        v = per.get(ch, {}).get(cond) if ch else None
        if v is None:
            ax.axis("off")
            continue
        im = ax.pcolormesh(v["ft"], v["f"], v["db"], cmap="bwr",
                           vmin=-cfg.ersp_vlim, vmax=cfg.ersp_vlim, shading="auto")
        ax.axvline(0, color="k", lw=1.0)
        ax.set_xlim(w0, w1)
        ax.set_box_aspect(1)
        ax.set_title(f"{titles.get(ch, ch)}  (n={v['n']})", fontsize=8.5, color=INK)
        ax.tick_params(labelsize=6.5, colors=MUTED, length=2)
    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
        cb.set_label("dB vs baseline", fontsize=8)
    kind = "analog inputs" if label.startswith("analog") else f"tetrode {label}"
    fig.suptitle(f"{pat}   {kind}   {cond}   0-{cfg.micro_ersp_fmax:.0f} Hz   ·   "
                 f"{subtitle}", fontsize=9.5, color=INK)
    fig.savefig(png, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def run(pat, reref="shaft_mean", do_notch=True, force=False, do_ersp=False, do_psd=True):
    import pandas as pd
    pre = cfg.MM_PRESETS[pat]
    tsv = os.path.join(ROOT, "outputs", pat, f"{pre['pat_name']}_MM_trials.tsv")
    if not os.path.exists(tsv):
        raise SystemExit(f"no trial table at {tsv}\n"
                         f"  run:  python .\\10_mm_triggers.py --patient {pat}")
    tr = pd.read_csv(tsv, sep="\t")
    print(f"\n=== {pat} ({pre['pat_name']})  micro   {len(tr)} trials   "
          f"reref={reref}  notch={'on' if do_notch else 'off'}")

    X, meta = extract_lfp(pat, pre, force=force)
    fs, names, t0 = meta["fs"], list(meta["names"]), float(meta["t0"])

    tag = f"{reref}{'' if do_notch else '_nonotch'}"
    out_dir = os.path.join(ROOT, "outputs", pat, "micro", tag)
    os.makedirs(out_dir, exist_ok=True)

    # the PSD overview is the micro montage: the analog inputs are not neural and would
    # move its median and IQR, so they stay out of it (they are in every other output)
    micro_rows = lambda X_, names_: ([i for i, n in enumerate(names_) if not is_ainp(n)])
    if do_psd:
        mi = micro_rows(X, names)
        psd_figure(X[mi], fs, [names[i] for i in mi], out_dir, pat, "psd_raw")

    X, names, dropped = rereference(X, names, reref)
    if dropped:
        print(f"  reref '{reref}': {len(dropped)} reference contacts are now identically "
              f"zero and are dropped: {', '.join(dropped)}")

    audit = []
    if do_notch:
        X, audit = apply_lm_notch(X, fs, names, pat, pre)
        print(f"  notch to {cfg.micro_notch_fmax:.0f} Hz: {len(audit)} harmonics acted on")
        if audit:
            import csv
            ap = os.path.join(out_dir, f"{pre['pat_name']}_notch_audit.tsv")
            keys = sorted({k for r in audit for k in r})
            with open(ap, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
                w.writeheader()
                w.writerows(audit)
            print(f"  wrote {ap}")
    if do_psd:
        mi = micro_rows(X, names)
        psd_figure(X[mi], fs, [names[i] for i in mi], out_dir, pat, "psd_clean")

    # a trial whose epoch spans a seam between .ns6 parts has a jump inside it: the parts
    # are not contiguous (4-6 s gaps, see lf_mm_io) and the cache is their concatenation
    parts, _tot = session_timeline(part_paths(pre))
    if not parts:
        parts = list(meta.get("parts", []))       # written by extract_lfp at cache time
        if not parts:
            print("  !! raw parts not reachable and the cache records no part layout: trials "
                  "that span a seam are NOT dropped (re-cache with --force when it is back)")
    x = crosses_splice(tr.onset_s.to_numpy(), cfg.time_window, parts)
    if x.any():
        print(f"  {int(x.sum())} trial(s) span a seam between .ns6 parts and are dropped: "
              + ", ".join(f"#{int(t)} {c} @{o:.1f}" for t, c, o in
                          zip(tr.trial[x], tr.condition[x], tr.onset_s[x])))
        tr = tr[~x].reset_index(drop=True)
    onsets = tr.onset_s.to_numpy() - t0           # cache starts at t0 on the session clock
    per, summary = {}, {}
    f = ft = None
    for i, ch in enumerate(names):
        x = X[i].astype(float)
        per[ch] = {}
        for c in cfg.COND_SHORT:
            on = onsets[tr.condition.to_numpy() == c]
            ep = epoch(x, fs, on, cfg.time_window)
            if len(ep) < 3:
                continue
            t, hz = hg_z(ep, fs)
            f, ft, db = ersp_db(ep, fs)
            per[ch][c] = dict(t=t, hg=hz.mean(axis=0), f=f, ft=ft, db=db, n=len(ep))
        if per[ch] and f is not None:
            band = (f >= cfg.broadband[0]) & (f <= cfg.broadband[1])
            resp = (ft >= 0.2) & (ft <= cfg.design_stim_s)
            summary[ch] = {c: float(np.nanmean(v["db"][np.ix_(band, resp)]))
                           for c, v in per[ch].items()}

    subtitle = (f"reref {reref}" + ("" if do_notch else ", NO notch")
                + (f", notched to {cfg.micro_notch_fmax:.0f} Hz" if do_notch else ""))
    tets = tetrodes(names)
    titles = {ch: chan_title(ch, pre) for ch in names}
    n_ain = sum(is_ainp(n) for n in names)
    print(f"  {len(names) - n_ain} contacts + {n_ain} analog inputs -> {len(tets)} groups")
    for label, grp in tets:
        plot_tetrode_hg(pat, label, grp, per,
                        os.path.join(out_dir, f"{label}_HG.png"), subtitle, titles)
        if do_ersp:
            for c in cfg.COND_SHORT:
                plot_tetrode_ersp(pat, label, grp, per, c,
                                  os.path.join(out_dir, f"{label}_ERSP_{c}.png"),
                                  subtitle, titles)

    p = os.path.join(out_dir, f"{pre['pat_name']}_MM_micro_broadband.tsv")
    conds = list(cfg.COND_SHORT)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("channel\t" + "\t".join(conds) + "\tmouth_minus_foot\n")
        for ch, v in summary.items():
            foot = np.nanmean([v.get(c, np.nan) for c in ("foot_left", "foot_right")])
            fh.write(ch + "\t" + "\t".join(f"{v.get(c, float('nan')):.3f}" for c in conds)
                     + f"\t{v.get('mouth', np.nan) - foot:.3f}\n")
    mf = lambda v: v.get("mouth", np.nan) - np.nanmean([v.get(c, np.nan)
                                                         for c in ("foot_left", "foot_right")])
    md = np.array([mf(v) for ch, v in summary.items() if not is_ainp(ch)], float)
    if len(md):
        print(f"  mouth minus foot over {len(md)} contacts: mean {np.nanmean(md):+.2f} dB  "
              f"median {np.nanmedian(md):+.2f}  {int((md > 0.5).sum())} above +0.5")
    ain = {ch: mf(v) for ch, v in summary.items() if is_ainp(ch)}
    if ain:
        print("  analog inputs, as recorded:  "
              + "   ".join(f"{titles[ch]} {v:+.2f} dB" for ch, v in ain.items()))
    print(f"  wrote {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="G-05")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--reref", default=cfg.micro_reref,
                    choices=["none", "first", "shaft_mean"])
    ap.add_argument("--no-notch", action="store_true")
    ap.add_argument("--no-psd", action="store_true")
    ap.add_argument("--ersp", action="store_true", help="also write per-condition ERSPs")
    ap.add_argument("--force", action="store_true", help="re-read the raw .ns6 parts")
    a = ap.parse_args()
    for p in (cfg.patient_ids if a.all else [a.patient]):
        try:
            run(p, reref=a.reref, do_notch=not a.no_notch, force=a.force,
                do_ersp=a.ersp, do_psd=not a.no_psd)
        except SystemExit as e:
            print(f"  !! {p}: {e}")
        except Exception as e:
            print(f"  !! {p} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
