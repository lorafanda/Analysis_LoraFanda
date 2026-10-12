#!/usr/bin/env python3
"""lf_microepi_ns6.py - the MicroEPI 30 kHz micro stream, and the trials inside it.

The .mat exports carry `dataMicroDown`, already downsampled. The .ns6 files are the full
30 kHz record, and they also carry the photodiode on an analog input, so a trial can be
cut from the micro stream WITHOUT syncing to the export's clock. That is what makes a
micro ERSP tree and spike sorting possible from one source.

Nothing here writes to DATARAW.

WHAT IS IN A FILE. 39 or 51 channels: 12 microwires per shaft (3 or 4 shafts) followed by
ainp1-3. The lower-case `m` on a shaft name is the micro marker (ADm, HADm, FPDm ...).

TETRODES. Wires group contiguously in fours: 1-4, 5-8, 9-12. The FIRST wire of each
tetrode (1, 5, 9) is wired as an optional hardware reference (R2), and wire 1 of the shaft
is additionally R1 - those contacts carry extra wiring and run noisier at times. They are
flagged, not dropped: `is_ref` marks them so a caller can exclude or audit them.
A shaft may hold 8 wires (2 tetrodes) instead of 12, so the grouping is derived.
"""
from __future__ import annotations

import os
import re
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from neo.rawio import BlackrockRawIO
from scipy.signal import butter, resample_poly, sosfiltfilt

RAW = "//nasac-m2.unige.ch/m-HumanNeuronLab/DATARAW/MICROEPI"

# folder, file span over the task, and the channel the photodiode is on. The session
# folder is named for when RECORDING started, which can be days before the task, and the
# NSx header t0 is 0.0034 s in every file - the spans come from the file mtimes bracketing
# the task time in the events-tsv name. `tmax` cuts a run short (G-05 ends with
# non-biological spikes). Polarity is decided from the data, not stored here.
SESSIONS = {
    "G-01": dict(pat="PAT_5515", folder="MicroEPI-G-01/raw_blackrock/20241127-161902",
                 stem="20241127-161902", files=list(range(273, 280)),
                 task="2024-11-28 15:52:08", trig="ainp1", tmax=None),
    "G-02": dict(pat="PAT_5533", folder="MicroEPI-G-02/raw_blackrock/20250115-175330",
                 stem="20250115-175330", files=list(range(222, 229)),
                 task="2025-01-16 13:56:00", trig="ainp1", tmax=None),
    "G-03": dict(pat="PAT_6619", folder="MicroEPI-G-03/raw_blackrock/20250307-093441",
                 stem="20250307-093441", files=list(range(55, 61)),
                 task="2025-03-07 14:59:34", trig="ainp1", tmax=None),
    "G-04": dict(pat="PAT_6704", folder="MicroEPI-G-04/raw_blackrock/20250402-170904",
                 stem="20250402-170904", files=list(range(1289, 1296)),
                 task="2025-04-07 14:55:53", trig="ainp1", tmax=None),
    "G-05": dict(pat="PAT_6684", folder="MicroEPI-G-05/raw_blackrock/20250618-145815",
                 stem="20250618-145815", files=list(range(10, 16)),
                 task="2025-06-18 15:39:19", trig="ainp1", tmax=1660.0),
    "G-06": dict(pat="PAT_6854", folder="MicroEPI-G-06/raw_blackrock/20260128-212210",
                 stem="20260128-212210", files=list(range(229, 235)),
                 task="2026-01-29 17:33:57", trig="ainp2", tmax=None),
}

# the three patients whose ainp2 is a microphone (G-06's ainp2 is its photodiode)
AUDIO_PATIENTS = ("G-03", "G-04", "G-05")
AUDIO_CHAN = "ainp2"
SPEECH_BAND = (300.0, 3000.0)   # formants, without the F0 range where mains harmonics sit


def is_ainp(name: str) -> bool:
    return "ainp" in str(name).lower()


def shaft_of(name: str) -> str:
    """'HADm7' -> 'HADm'. The trailing digits are the contact number."""
    return re.sub(r"\d+$", "", str(name))


def tetrodes(names):
    """[(shaft, k, [names of the 4 wires]), ...] in contact order, plus a ref flag dict.

    Wires group 1-4, 5-8, 9-12. `is_ref[name]` is 'R1' for wire 1 of a shaft, 'R2' for the
    first wire of every other tetrode, else ''. A trailing group of fewer than 4 wires is
    not a tetrode and is returned in `odd`.
    """
    order, by_shaft = [], {}
    for n in names:
        if is_ainp(n):
            continue
        s = shaft_of(n)
        if s not in by_shaft:
            by_shaft[s] = []
            order.append(s)
        by_shaft[s].append(n)
    groups, odd, is_ref = [], [], {}
    for s in order:
        # group by the CONTACT NUMBER, not by position in the list. Position breaks as
        # soon as a wire is missing: drop R1 and an 11-wire shaft regroups as 2-5, 6-9,
        # 10-12, straddling the real tetrode boundaries and sorting wires from two
        # different tetrodes together. The quads are a property of the electrode.
        num = {}
        for n in by_shaft[s]:
            m = re.search(r"(\d+)$", n)
            if m:
                num[n] = int(m.group(1))
        quads = {}
        for n, v in num.items():
            quads.setdefault((v - 1) // 4, []).append(n)
            if v % 4 == 1:                      # 1, 5, 9: the tetrode-leading contacts
                is_ref[n] = "R1" if v == 1 else "R2"
        for k in sorted(quads):
            quad = sorted(quads[k], key=lambda x: num[x])
            if len(quad) < 2:                   # a lone survivor is not a tetrode
                odd.extend(quad)
                continue
            groups.append((s, k, quad))
    for n in names:
        if not is_ainp(n):
            is_ref.setdefault(n, "")
    return groups, odd, is_ref


def reref(x, names, scheme="shaft_r1", drop_ref=True):
    """Re-reference the micro channels. Returns (x, names, dropped).

    `shaft_r1` (default) subtracts wire 1 of each shaft from every other wire on that
    shaft. That wire is R1, physically wired as a hardware reference, so this is the
    local-reference scheme the human microwire literature uses on Behnke-Fried and Dixi
    hybrid electrodes before spike sorting: it cancels the shaft's common pickup while
    staying local, where a whole-array average would mix in activity from other
    structures. R1 itself is all zeros afterwards and is dropped.

    `tetrode_car` instead subtracts the mean of each tetrode's own four wires. It survives
    a dead or saturated R1, but any spike shared across the four wires is attenuated,
    which is the signal a tetrode exists to capture - prefer it only when R1 is unusable.

    `none` leaves the data alone (the hardware reference may already be applied).
    """
    x = np.asarray(x)
    groups, _odd, ref_of = tetrodes(names)
    idx = {n: i for i, n in enumerate(names)}
    out = x.copy()
    dropped = []
    if scheme == "none":
        return out, list(names), dropped
    if scheme == "shaft_r1":
        r1 = {n: k for n, k in ref_of.items() if k == "R1"}
        for shaft in {shaft_of(n) for n in r1}:
            ref_name = next(n for n in r1 if shaft_of(n) == shaft)
            if ref_name not in idx:
                continue
            cols = [idx[n] for n in names if not is_ainp(n) and shaft_of(n) == shaft]
            out[:, cols] -= x[:, idx[ref_name]][:, None]
            dropped.append(ref_name)
    elif scheme == "tetrode_car":
        for _s, _k, quad in groups:
            cols = [idx[n] for n in quad if n in idx]
            if len(cols) < 2:
                continue
            out[:, cols] -= x[:, cols].mean(axis=1, keepdims=True)
    else:
        raise SystemExit(f"unknown reref scheme {scheme!r}")
    if drop_ref and dropped:
        keep = [i for i, n in enumerate(names) if n not in dropped]
        out = out[:, keep]
        names = [names[i] for i in keep]
    return out, list(names), dropped


def _paths(patient, files=None):
    S = SESSIONS[patient]
    use = files if files is not None else S["files"]
    out = []
    for i in use:
        p = f"{RAW}/{S['folder']}/{S['stem']}-{i:03d}"
        if os.path.exists(p + ".ns6"):
            out.append((i, p))
    if not out:
        raise SystemExit(f"{patient}: none of the files {list(use)} exist")
    return S, out


def channel_names(patient, files=None):
    """The channel list, from the header alone - no samples are read.

    Asking read_ns6 for this pulls a whole 0.86 GB file off the share to look at a list of
    39 strings; parse_header does not touch the data block.
    """
    _S, paths = _paths(patient, files)
    r = BlackrockRawIO(filename=paths[0][1], nsx_to_load=6)
    r.parse_header()
    return [str(v) for v in r.header["signal_channels"]["name"]]


def read_ns6(patient, files=None, want=None, decimate_to=None, band=None, verbose=True):
    """Concatenate the task files. Returns dict(x, t, names, fs, bounds, is_ref).

    `want` is a list of channel names (default: every channel). `decimate_to` resamples to
    that rate with an anti-aliased polyphase filter - the ERSP only reaches 400 Hz, and
    48 channels at 30 kHz is ~5 GB of int16 for half an hour. `band` band-passes at the
    FULL rate first, which is what the speech band needs.
    """
    S, paths = _paths(patient, files)
    xs, ts, bounds, off = [], [], [], 0.0
    names = fs_out = None
    for i, p in paths:
        if verbose:
            print(f"  reading {os.path.basename(p)}.ns6 ...", flush=True)
        r = BlackrockRawIO(filename=p, nsx_to_load=6)
        r.parse_header()
        sig = r.header["signal_channels"]
        nm = [str(v) for v in sig["name"]]
        fs = float(sig["sampling_rate"][0])
        idx = [j for j, n in enumerate(nm) if (want is None or n in want)]
        if not idx:
            raise SystemExit(f"{patient}: none of {want} in {nm}")
        n = r.get_signal_size(0, 0)
        raw = r.get_analogsignal_chunk(block_index=0, seg_index=0, i_start=0, i_stop=n,
                                       channel_indexes=np.asarray(idx))
        x = r.rescale_signal_raw_to_float(raw, dtype="float32",
                                          channel_indexes=np.asarray(idx))
        if band is not None:
            sos = butter(4, list(band), btype="band", fs=fs, output="sos")
            x = sosfiltfilt(sos, x.astype("float64"), axis=0).astype("float32")
        if decimate_to and decimate_to < fs:
            g = int(round(fs / decimate_to))
            if abs(fs / g - decimate_to) > 1e-6:
                raise SystemExit(f"{fs:.0f} Hz does not divide to {decimate_to} Hz")
            x = resample_poly(x, 1, g, axis=0).astype("float32")   # anti-aliased
            fs = fs / g
        names = [nm[j] for j in idx] if names is None else names
        fs_out = fs
        t = np.arange(x.shape[0], dtype="float64") / fs
        xs.append(x)
        ts.append(off + t)
        bounds.append((off, i))
        off += float(t[-1])
    x = np.concatenate(xs, axis=0)
    t = np.concatenate(ts)
    tmax = SESSIONS[patient].get("tmax")
    if tmax is not None:
        m = t <= tmax
        x, t = x[m], t[m]
    _, _, is_ref = tetrodes(names)
    return dict(x=x, t=t, names=names, fs=fs_out, bounds=bounds, is_ref=is_ref, pat=S["pat"])


def detect_cues(trig, t, fs, min_pulse=0.2, flip=False, block_s=60.0, base_pct=20.0):
    """[(onset, end), ...] in seconds, from the photodiode trace.

    The diode's RESTING LEVEL drifts between files (G-04 rests near 430 in some and 300 in
    others), so a threshold on the raw trace reads a whole file as "high" and the pulses it
    finds are the gaps between stimuli. The resting level is subtracted first.

    That level is a LOW PERCENTILE of a long block, not a median of a short one. The
    stimulus is not always a small part of its trial: audio runs 3.09 s on against a 5.97 s
    interval, a 52% duty cycle, so a 10 s block median sits INSIDE the stimulus and
    subtracts the pulse away. With a 10 s median only 100 of 160 trials were found and the
    ones that survived were nearly all picture (1.00 s on, 28% duty); p20 over 60 s finds
    all 160 on both G-03 and G-04.

    Polarity comes from the data, never from config 'flip': that field describes the .mat
    export and disagrees with the raw channel (G-03 is flip=True yet reads straight).
    Whichever way the stimulus pushes the diode off rest is the bigger excursion.
    Returns (cues, inverted).
    """
    g = np.asarray(trig, dtype="float64").ravel()
    blk = max(1, int(block_s * fs))
    base = np.repeat(np.array([np.percentile(g[j:j + blk], base_pct)
                               for j in range(0, len(g), blk)]), blk)[:len(g)]
    r = g - base
    p_hi, p_lo = float(np.percentile(r, 99)), float(np.percentile(r, 1))
    inverted = bool(flip) or abs(p_lo) > abs(p_hi)
    if inverted:
        r = -r
    # The pulse height is the median of the top quartile, NOT a high percentile. G-02
    # carries two +-700 transients; they set p99, the threshold became half an artifact,
    # and the detector found 3 cues in 35 minutes instead of ~160. A median over the upper
    # mode ignores a handful of outliers, and with any duty cycle above ~25% the top
    # quartile is the stimulus plateau itself.
    hi_gate = float(np.percentile(r, 75))
    top = r[r > hi_gate]
    amp = float(np.median(top)) if top.size else 0.0
    if amp <= 1e-9:
        return [], inverted
    b = r > 0.5 * amp
    w = max(1, int(round(min_pulse * fs)))
    if w > 1:
        # the box is centred, so a step reads as high only w/2 later - roll it back
        b = np.roll(np.convolve(b.astype("float32"), np.ones(w, "float32") / w, "same")
                    > 0.9, -(w // 2))
    up = np.flatnonzero(b[1:] & ~b[:-1]) + 1
    dn = np.flatnonzero(~b[1:] & b[:-1]) + 1
    cues = []
    for s in up:
        j = int(np.searchsorted(dn, s))
        if j >= len(dn):                     # the pulse runs past the end of the data
            continue
        on, end = float(t[s]), float(t[dn[j]])
        if not cues or on - cues[-1][0] > 0.25:
            cues.append((on, end))
    return cues, inverted


def load_140_trials(patient, cfg=None):
    """140's trials for this patient, in the EXPORT's clock (seconds).

    These are the canonical trials: 140 read them from the .mat photodiode with the
    preset's time_range / pd_blocks / invalid_trials / fake_trials applied, and they carry
    the condition, the accuracy and the stimulus duration. They are far better than
    re-detecting on the .ns6, which misses faded pulses (122 of ~158 cues on G-03).

    Returns a list of dicts with onset, duration, trial_end (export seconds) and condition.
    """
    import csv
    import glob as _glob
    if cfg is None:
        import config as cfg                                     # noqa: F401
    pat = SESSIONS[patient]["pat"]
    roots = []
    pr = getattr(cfg, "MICROEPI_MAT_PRESETS", {}).get(patient)
    if pr:
        roots.append(os.path.join(os.path.dirname(str(pr["data_dir"])), "prep0"))
    # a retired preset has no .mat route: G-05 runs as PAT_6684 off its TRC, and its trials
    # sit under the PATH_OVERRIDES base_root instead (task_FBM/data_LM/prep0)
    ov = getattr(cfg, "PATH_OVERRIDES", {}).get(pat, {})
    if ov.get("base_root"):
        br = str(ov["base_root"])
        roots += _glob.glob(os.path.join(br, "prep0"))
        roots += _glob.glob(os.path.join(br, "*", "prep0"))
    hits = []
    for prep in roots:
        hits += [p for p in _glob.glob(os.path.join(prep, f"{pat}_*triggerPD*.tsv"))
                 if not p.endswith(".bak")]
    if not hits:
        raise SystemExit(f"{patient} ({pat}): no triggerPD tsv under {roots or '(no root)'} "
                         f"- run 140 --pd first")
    newest = {}                                   # one file per condition: the latest date
    for p in hits:
        b = os.path.basename(p)
        cond = b.split("_LM_")[1].split("__")[0] if "_LM_" in b else b
        if cond not in newest or b > os.path.basename(newest[cond]):
            newest[cond] = p
    out = []
    for cond, p in sorted(newest.items()):
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                try:
                    out.append(dict(onset=float(row["onset"]),
                                    duration=float(row.get("onset_duration") or "nan"),
                                    trial_end=float(row["trial_end"]) if row.get("trial_end") else float("nan"),
                                    sample=int(float(row["sample"])) if row.get("sample") else -1,
                                    cond=cond,
                                    cond_name=row.get("condition_name", cond),
                                    accuracy=row.get("resp_accuracy", ""),
                                    trial_idx=row.get("trial_idx", "")))
                except (TypeError, ValueError):
                    continue
    out.sort(key=lambda d: d["onset"])
    return out


def sync_to_ns6(mat_onsets, ns6_cues, tol=0.08, n_anchor=25):
    """Map export-clock seconds onto .ns6 seconds. Returns (a, b, info) for ns6 = a*mat + b.

    The two streams record the SAME physical photodiode, so their event trains differ only
    by a start offset and whatever the two acquisition clocks drift apart by. The offset is
    found by trying anchor pairings and keeping the one that matches the most events, then
    `a` and `b` come from a least-squares fit on the matched pairs. `a` is the clock ratio
    and should sit within a few parts per million of 1 - if it does not, or if the residual
    is more than a sample or two, the mapping is not trustworthy and the caller should stop
    rather than cut trials in the wrong place.
    """
    A = np.asarray(sorted(float(x) for x in mat_onsets), dtype="float64")
    B = np.asarray(sorted(float(c[0] if isinstance(c, (tuple, list)) else c)
                          for c in ns6_cues), dtype="float64")
    if len(A) < 3 or len(B) < 3:
        raise SystemExit("not enough events to sync")

    def _hits(scale, shift):
        """Which trials land on a pulse, vectorised - the grid below evaluates this ~20k
        times, and a Python loop over the trials there costs minutes instead of seconds."""
        tgt = scale * A + shift
        j = np.searchsorted(B, tgt)
        ok = np.zeros(len(A), dtype=bool)
        nearest = np.empty(len(A))
        for lo in (-1, 0):
            kk = np.clip(j + lo, 0, len(B) - 1)
            d = np.abs(B[kk] - tgt)
            take = (d <= tol) & ~ok
            nearest[take] = B[kk][take]
            ok |= d <= tol
        return ok, nearest

    # SEARCH the clock ratio, do not fit it from whatever matched first. Fitting it that
    # way is circular - the first matches come from one short stretch of the session, a
    # slope fitted over a short span is badly constrained, and extrapolating it across
    # ~1600 s puts the far end 80 ms out, which is the whole tolerance window. That is how
    # G-01 ended up at +49.8 ppm where the same hardware reads -17.5 ppm, with only the
    # first block matching. The grid is coarse because what matters is drift across the
    # session: 10 ppm over 1600 s is 16 ms, comfortably inside tol.
    best = (0, 1.0, 0.0)
    for ppm in range(-150, 151, 10):
        scale = 1.0 + ppm * 1e-6
        for i in range(min(n_anchor, len(A))):
            for jj in range(min(n_anchor, len(B))):
                shift = B[jj] - scale * A[i]
                n = int(_hits(scale, shift)[0].sum())
                if n > best[0]:
                    best = (n, scale, shift)
    hit, _scale, shift = best
    ok, nearest = _hits(_scale, shift)
    pairs = list(zip(A[ok], nearest[ok]))
    if hit < 3:
        raise SystemExit("the two photodiode trains do not line up - sync failed")
    pa = np.array([p[0] for p in pairs])
    pb = np.array([p[1] for p in pairs])
    a, b = (np.polyfit(pa, pb, 1) if len(pairs) >= 2 else (1.0, shift))
    resid = pb - (a * pa + b)
    info = dict(matched=hit, n_mat=len(A), n_ns6=len(B), shift=shift, ratio=float(a),
                resid_ms=float(np.max(np.abs(resid)) * 1e3),
                resid_rms_ms=float(np.sqrt(np.mean(resid ** 2)) * 1e3))
    return float(a), float(b), info


def _best_offset(onsets, B_on, a, tol, anchors=40):
    """The offset b that places the most of `onsets` on a pulse, with the ratio a fixed.

    Returns (b, [onsets it matched]) or None. Only the offset is searched: the clock ratio
    belongs to the hardware and does not change at a file join, so re-fitting it on a
    handful of trials would just chase noise.
    """
    A = np.asarray(sorted(onsets), dtype="float64")
    if len(A) == 0 or len(B_on) == 0:
        return None
    best = None
    for i in range(min(anchors, len(A))):
        for j in range(len(B_on)):
            b = float(B_on[j] - a * A[i])
            tgt = a * A + b
            k = np.searchsorted(B_on, tgt)
            ok = np.zeros(len(A), dtype=bool)
            for lo in (-1, 0):
                kk = np.clip(k + lo, 0, len(B_on) - 1)
                ok |= np.abs(B_on[kk] - tgt) <= tol
            n = int(ok.sum())
            if best is None or n > best[0]:
                best = (n, b, A[ok])
    if best is None or best[0] == 0:
        return None
    return best[1], list(best[2])


def trials_on_ns6(patient, cues=None, files=None, tol=0.08, cfg=None, verbose=True,
                  min_seg=12, max_seg_rms=20.0, max_segments=3, report=False):
    """140's trials, each carrying the time of ITS OWN pulse in the .ns6 stream.

    The trial LIST comes from 140 (curated: time_range, pd_blocks, invalid_trials,
    fake_trials, plus condition and accuracy). The trial TIME comes from the .ns6
    photodiode, matched pulse to pulse. Nothing is mapped through a fitted clock, so the
    ~8-10 ms the two systems disagree by - they threshold the same slow screen ramp at
    different levels - never reaches the epochs. The fit is still reported, because a
    ratio far from 1 or a trial that fails to match means the two records are not the same
    session and the caller should stop.

    Each trial gains t_on / t_end (seconds into the concatenated .ns6) and ns6_dur.
    Trials with no matching pulse are returned too, with t_on None, and counted.
    """
    tr = load_140_trials(patient, cfg=cfg)
    if cues is None:
        cues, _ = cue_times(patient, files=files, verbose=verbose)
    a, b, info = sync_to_ns6([q["onset"] for q in tr], cues)
    B_on = np.array(sorted(c[0] for c in cues))
    by_on = {c[0]: c[1] for c in cues}

    # SEGMENTS. The export is a concatenation of .mat files (f0001, f0002, ...) and its
    # clock jumps where they join, so one straight line can only follow one side of the
    # join. On G-01, G-02, G-05 and G-06 that left exactly the picture block matched - the
    # first block - and dropped audio and reading entirely. The ratio is a property of the
    # two crystals and stays fixed; only the offset moves, so each further pass keeps `a`
    # and searches for the offset that explains the most of what is still unmatched.
    # A segment has to EARN its place. Unconstrained, this kept finding offsets that lined
    # a handful of trials up with whatever cue happened to be near, and the overall jitter
    # rose from 10 ms to 40 ms while the match count went up - more trials located, worse
    # located. A real join shows up as a large block of trials that lands tightly; so a
    # pass is kept only if it explains at least `min_seg` trials AND its own residual is
    # tighter than `max_seg_rms`. The export is a concatenation of a handful of .mat
    # files, so `max_segments` stays small on purpose.
    fit = {}                                   # onset -> offset b to use for that trial
    todo = [q["onset"] for q in tr]
    offsets, rejected = [], 0
    for _ in range(max_segments):
        if not todo:
            break
        cand = _best_offset(todo, B_on, a, tol)
        if cand is None:
            break
        b_seg, hit_on = cand
        if len(hit_on) < min_seg:
            rejected += len(hit_on)
            break
        res = []
        for o in hit_on:
            t = a * o + b_seg
            k = int(np.searchsorted(B_on, t))
            near = [B_on[x] for x in (k - 1, k) if 0 <= x < len(B_on)]
            if near:
                res.append(min(near, key=lambda v: abs(v - t)) - t)
        rms = float(np.sqrt(np.mean(np.square(res))) * 1e3) if res else 1e9
        if rms > max_seg_rms:
            rejected += len(hit_on)
            break
        offsets.append((b_seg, len(hit_on), rms))
        for o in hit_on:
            fit[round(o, 6)] = b_seg
        todo = [o for o in todo if round(o, 6) not in fit]
    if not offsets:
        offsets = [(b, 0, float("nan"))]

    out, miss, dev = [], 0, []
    for q in tr:
        b_q = fit.get(round(q["onset"], 6), b)
        tgt = a * q["onset"] + b_q
        j = int(np.searchsorted(B_on, tgt))
        hit = None
        for jj in (j - 1, j):
            if 0 <= jj < len(B_on) and abs(B_on[jj] - tgt) <= tol:
                hit = float(B_on[jj])
                break
        r = dict(q)
        r["t_raw"] = hit
        r["t_end_raw"] = by_on.get(hit) if hit is not None else None
        r["ns6_dur"] = (r["t_end_raw"] - r["t_raw"]) if hit is not None else float("nan")
        if hit is None:
            miss += 1
        else:
            dev.append(tgt - hit)
        out.append(r)

    # THE CONSTANT BIAS. A threshold crossing on a screen's luminance ramp is late on the
    # rise and early on the fall, so the raw .ns6 pulse starts ~80 ms after 140's onset and
    # ends ~80 ms before its offset - the same in every condition and both patients, which
    # is what makes it a bias and not jitter. The median disagreement is removed here, so
    # t_on carries 140's definition of onset while keeping the .ns6 pulse's own timing
    # (no fitted clock in the epochs). What is left is the per-trial scatter, reported as
    # resid_rms_ms. The stimulus END is taken as 140's duration, which is the reference the
    # rest of the analysis uses; ns6_dur keeps the uncorrected measurement for comparison.
    bias = float(np.median(dev)) if dev else 0.0
    for r in out:
        r["t_on"] = (r["t_raw"] + bias) if r["t_raw"] is not None else None
        r["t_end"] = (r["t_on"] + r["duration"]
                      if r["t_on"] is not None and np.isfinite(r["duration"]) else None)
    info["unmatched"] = miss
    info["bias_ms"] = bias * 1e3
    info["jitter_ms"] = float(np.std(dev) * 1e3) if dev else float("nan")
    info["segments"] = offsets
    info["rejected"] = rejected
    if verbose:
        seg = (f"{len(offsets)} clock segments ("
               + "/".join(f"{n}@{r:.0f}ms" for _b, n, r in offsets) + ")"
               if len(offsets) > 1 else
               f"one clock segment ({offsets[0][1]} trials)")
        print(f"  {len(out) - miss}/{len(out)} trials located in the .ns6 | "
              f"{(info['ratio']-1)*1e6:+.1f} ppm | {seg} | bias {info['bias_ms']:+.0f} ms "
              f"removed, {info['jitter_ms']:.1f} ms jitter left"
              + (f" | {rejected} trials left out as too loose to trust" if rejected else ""))

    if report:
        # Do the two sets even describe the same events? Intervals are the fingerprint:
        # they are invariant to any offset and, at ppm level, to the clock ratio too
        # (6 s x 50 ppm is 0.3 ms). If the trial intervals and the cue intervals are the
        # same sequence, the events correspond and only the mapping is wrong. If they are
        # not, the .ns6 window is showing a different set of events and no fit can help.
        A_on = np.array(sorted(q["onset"] for q in tr))
        dA, dB = np.diff(A_on), np.diff(B_on)
        print(f"  trial intervals: n={len(dA)} median {np.median(dA):.2f}s  "
              f"quartiles {np.percentile(dA,25):.2f}/{np.percentile(dA,75):.2f}")
        print(f"  cue   intervals: n={len(dB)} median {np.median(dB):.2f}s  "
              f"quartiles {np.percentile(dB,25):.2f}/{np.percentile(dB,75):.2f}")
        print("  first 18 trial intervals: " + " ".join(f"{v:.2f}" for v in dA[:18]))
        print("  first 18 cue   intervals: " + " ".join(f"{v:.2f}" for v in dB[:18]))
        # the best run of consecutive intervals that agree, anywhere in either sequence
        bestrun = (0, -1, -1)
        for i in range(len(dA)):
            for j in range(len(dB)):
                k = 0
                while (i + k < len(dA) and j + k < len(dB)
                       and abs(dA[i + k] - dB[j + k]) < 0.05):
                    k += 1
                if k > bestrun[0]:
                    bestrun = (k, i, j)
        print(f"  longest matching run of intervals: {bestrun[0]} "
              f"(trial {bestrun[1]} vs cue {bestrun[2]}) - a real correspondence runs "
              f"into the tens; a handful means these are different events")

        # where the export and the .ns6 part company: the residual of every trial against
        # the FIRST segment's line, in time order, with the condition beside it. A single
        # step means one join; a sawtooth means the export is dropping samples throughout.
        b0 = offsets[0][0]
        print(f"  {'onset_s':>10} {'cond':<8} {'resid_ms':>9}  segment")
        for q in sorted(tr, key=lambda z: z["onset"]):
            t = a * q["onset"] + b0
            k = int(np.searchsorted(B_on, t))
            near = [B_on[x] for x in (k - 1, k) if 0 <= x < len(B_on)]
            d = (min(near, key=lambda v: abs(v - t)) - t) * 1e3 if near else float("nan")
            which = next((i for i, (bb, _n, _r) in enumerate(offsets)
                          if abs(fit.get(round(q["onset"], 6), b0) - bb) < 1e-9), -1)
            print(f"  {q['onset']:10.2f} {q['cond']:<8} {d:9.1f}  "
                  f"{which if round(q['onset'],6) in fit else '-'}")
    return out, info


def cue_times(patient, files=None, min_pulse=0.2, verbose=True):
    """The trial times for a patient, read from the photodiode channel alone (one channel
    of fifty-one, so this is cheap compared with loading the micros)."""
    S = SESSIONS[patient]
    d = read_ns6(patient, files, want=[S["trig"]], verbose=verbose)
    cues, inv = detect_cues(d["x"][:, 0], d["t"], d["fs"], min_pulse=min_pulse)
    if verbose and cues:
        dur = float(np.median([e - o for o, e in cues]))
        print(f"  {len(cues)} cues on {S['trig']} "
              f"({'inverted' if inv else 'straight'}), median {dur:.2f} s long")
    return cues, d
