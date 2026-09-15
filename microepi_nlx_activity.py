#!/usr/bin/env python3
"""
microepi_nlx_activity.py - what was actually going on while the Neuralynx was recording?

v2 of microepi_nlx_durations.py. That one answers "was it recording"; this one answers
"and what was it recording" - is there signal or mains hum, were there any event markers,
did the photodiode line ever fire, were spikes being detected.

Four things are read, three of them nearly free:

  MARKERS   .nev holds the TTL events. The Pegasus log also holds the -SetNamedTTLEvent
            lines that DECLARE which port and bit carries which name, so declared and
            fired can be compared - a marker line that was configured and never pulsed
            looks identical to one that was never configured, unless you read both.
            The pulse train is then characterised by its own timing: a median interval
            of 16.7 ms at 60 pulses a second is a screen refresh picked up by a
            photodiode; one every 4.5 s, irregular, is a trial marker.

  RECORDING Starting / Stopping Recording events, and Ref Changed - a reference swapped
            mid-session changes what every later sample means, so it is drawn.

  SPIKES    .nse files are already on disk, one per micro channel, 112 bytes a spike
            with the timestamp first. Reading ~64 timestamps per file by index gives the
            cumulative-count curve, and its slope is the spike rate over time - the
            whole session's unit activity for a few hundred bytes a channel.

  SIGNAL    the only part that reads samples. Short windows are taken at intervals
            through each macro block and turned into: RMS in uV, saturation fraction,
            mains (50 Hz) share of 1-200 Hz power, railway (16.7 Hz) share, and the
            low-over-high ratio that separates a 1/f-ish LFP from flat noise. Windows
            are 2 s of 2 kHz macro - eight records, 8 KB - so a whole session costs a
            few megabytes rather than the hundreds of gigabytes it holds.

Every threshold that turns those numbers into a word ("mains", "flat", "LFP-like") is in
LABEL_RULES below, and every raw number is in the probes CSV, so a label can be argued
with without re-reading the share.

Outputs, per MicroEPI-B-## folder, into outputs/microepi_nlx/
  <folder>_activity.png    the v1 timeline with signal, marker and spike lanes under it
  <folder>_markers.csv     one row per declared or fired TTL name, per session
  <folder>_probes.csv      one row per signal window, with every metric
  <folder>_spikes.csv      one row per micro channel
  _activity_cache.json     so the figure can be redrawn without re-reading the share

Usage
    python microepi_nlx_activity.py                 # everything
    python microepi_nlx_activity.py --folder B-02   # one folder
    python microepi_nlx_activity.py --no-signal     # markers and spikes only, seconds
    python microepi_nlx_activity.py --use-cache     # redraw from the last scan
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

# The .ncs reader, the share-safe writers and the timeline geometry all live in v1;
# importing by path rather than by name keeps it working whatever the CWD is.
_spec = importlib.util.spec_from_file_location("nlxdur", HERE / "microepi_nlx_durations.py")
V1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V1)

HDR_BYTES, REC_BYTES, SAMP_PER_REC = V1.HDR_BYTES, V1.REC_BYTES, V1.SAMP_PER_REC
NEV_REC = 184
fmt_hms, fmt_hm, short = V1.fmt_hms, V1.fmt_hm, V1.short

# Swiss grid is 50 Hz; the Swiss railway runs at 16.7 Hz and its comb lands on top of the
# EEG band in some rooms. Harmonics of 16.7 that coincide with a mains harmonic are left
# out of the railway band so the two are not credited with the same power.
MAINS_HZ = 50.0
RAIL_HZ = 16.667
BROAD = (1.0, 200.0)

LABEL_RULES = """
  saturated    more than 1% of samples at 99.5% of the AD range - amplifier railing
               (tested first: a pinned channel is flat, and would read as empty)
  empty        rms < 1 uV, or fewer than 3 distinct sample values - nothing connected
  mains        50 Hz + harmonics hold more than half of 1-200 Hz power
  railway      16.7 Hz comb holds more than a fifth of it
  LFP-like     low band (1-8 Hz) at least 3x the high band (60-120 Hz), 5-500 uV rms
  unclear      none of the above
"""


# ------------------------------------------------------------------------- .nev events


def read_nev(path: Path) -> list[tuple[int, int, int, str]]:
    """(timestamp us, event id, ttl value, event string) for one .nev file."""
    n = (path.stat().st_size - HDR_BYTES) // NEV_REC
    if n <= 0:
        return []
    with path.open("rb") as fh:
        fh.seek(HDR_BYTES)
        buf = fh.read(n * NEV_REC)
    out = []
    for i in range(n):
        r = buf[i * NEV_REC:(i + 1) * NEV_REC]
        ts, eid, ttl = struct.unpack("<Qhh", r[6:18])
        out.append((ts, eid, ttl, r[56:184].split(b"\x00")[0].decode("latin-1")))
    return out


DECL_RE = re.compile(r'SetNamedTTLEvent "([^"]+)" (\d+) (\d+) "([^"]+)"')
REF_RE = re.compile(r"Ref Changed (\d+->\d+)")


def declared_ttl(sdir: Path) -> list[dict]:
    """The marker lines Pegasus was CONFIGURED with, from its own log.

    Without this a line that was set up and never pulsed is indistinguishable from one
    nobody ever configured - which is the whole question about the photodiode.
    """
    log = sdir / "PegasusLogFile.txt"
    if not log.exists():
        return []
    seen: dict[tuple[int, int], str] = {}
    for _dev, port, bit, name in DECL_RE.findall(log.read_text(errors="replace")):
        seen[(int(port), int(bit))] = name
    return [{"port": p, "bit": b, "name": nm} for (p, b), nm in sorted(seen.items())]


def describe_train(times: list[float], burst_gap: float) -> dict:
    """What KIND of marker line is this - a frame clock, or trial markers?

    A photodiode watching a 60 Hz screen pulses every 16.7 ms with almost no jitter; a
    trial marker lands every few seconds and wanders. Median interval plus the fraction
    of intervals close to it separates the two without being told which to expect.
    """
    t = np.asarray(sorted(times), float)
    if len(t) < 2:
        return {"n": len(t), "median_isi": None, "regular_frac": 0.0,
                "verdict": "single pulse" if len(t) == 1 else "never fired", "bursts": []}
    dt = np.diff(t)
    dt = dt[dt > 0]
    med = float(np.median(dt))
    regular = float(np.mean(np.abs(dt - med) < 0.02 * med))

    edges = np.where(np.diff(t) > burst_gap)[0]
    starts = np.r_[0, edges + 1]
    stops = np.r_[edges, len(t) - 1]
    bursts = [
        {"start": float(t[a]), "end": float(t[b]), "n": int(b - a + 1),
         "median_isi": float(np.median(np.diff(t[a:b + 1]))) if b > a else None}
        for a, b in zip(starts, stops)
    ]

    if med < 0.05 and regular > 0.8:
        verdict = f"frame-locked, {1 / med:.0f} pulses/s - photodiode-like"
    elif regular > 0.8:
        verdict = f"regular, every {med:.2f}s"
    else:
        verdict = f"irregular, median {med:.2f}s - trial markers"
    return {"n": int(len(t)), "median_isi": med, "regular_frac": regular,
            "verdict": verdict, "bursts": bursts}


def session_markers(sdir: Path, burst_gap: float) -> dict:
    """Every .nev in one session, split into marker trains, recording edges and refs."""
    events = []
    for f in sorted(sdir.glob("*.nev")):
        events += read_nev(f)
    events.sort()

    named: dict[str, list[float]] = defaultdict(list)
    rec_edges, ref_changes = [], []
    auto = 0
    for ts, _eid, _ttl, s in events:
        t = ts / 1e6
        if s.startswith("TTL Input"):
            auto += 1                      # the automatic partner of every named pulse
        elif s in ("Starting Recording", "Stopping Recording"):
            rec_edges.append({"t": t, "kind": s.split()[0].lower()})
        elif REF_RE.search(s):
            ref_changes.append({"t": t, "text": s})
        else:
            named[s].append(t)

    decl = declared_ttl(sdir)
    trains = {name: describe_train(ts, burst_gap) for name, ts in named.items()}
    for d in decl:                          # declared but silent still gets a row
        trains.setdefault(d["name"], describe_train([], burst_gap))
    return {"declared": decl, "trains": trains, "rec_edges": rec_edges,
            "ref_changes": group_ref_changes(ref_changes), "n_events": len(events),
            "n_auto_ttl": auto}


def group_ref_changes(raw: list[dict]) -> list[dict]:
    """One row per reference change, not one per channel.

    Pegasus announces a reference switch separately for every channel it touches, so
    EL043's six changes arrive as 160 events. Counting the events says the reference
    moved 160 times, which is not what happened; they are grouped by the second.
    """
    by_second: dict[int, list[dict]] = defaultdict(list)
    for r in raw:
        by_second[int(round(r["t"]))].append(r)
    out = []
    for sec, group in sorted(by_second.items()):
        moves = Counter(m.group(1) for m in
                        (REF_RE.search(g["text"]) for g in group) if m)
        out.append({
            "t": min(g["t"] for g in group),
            "n_channels": len(group),
            "transition": ", ".join(k for k, _ in moves.most_common(3)),
            "example": group[0]["text"],
        })
    return out


# -------------------------------------------------------------------------- .nse spikes


def spike_curve(path: Path, n_probes: int) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Cumulative spike count against time, from ~n_probes timestamps.

    Spike timestamps are stored in order, so the i-th record IS the i-th spike: reading
    a scattering of them by index gives the whole cumulative curve for a few hundred
    bytes. Its slope is the firing rate.
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(HDR_BYTES).split(b"\x00", 1)[0].decode("latin-1")
            g = lambda k: (re.search(r"^-%s (.*)$" % k, head, re.M) or [None, ""])[1].strip()
            rs = int(g("RecordSize") or 112)
            # The spike entity carries its own rate and threshold, and it is NOT safe to
            # infer either from the name: EL043 calls its 32 kHz micro spike channels
            # lead_A_L-101 while lead_A_L-1 is a 2 kHz macro contact.
            meta = {"fs": float(g("SamplingFrequency") or 0),
                    "threshold": g("ThreshVal"), "entity": g("AcqEntName")}
            n = (path.stat().st_size - HDR_BYTES) // rs
            if n < 2:
                return None
            idx = np.unique(np.linspace(0, n - 1, min(n, n_probes)).astype(int))
            ts = []
            for i in idx:
                fh.seek(HDR_BYTES + int(i) * rs)
                ts.append(struct.unpack("<Q", fh.read(8))[0] / 1e6)
    except OSError:
        return None
    return np.asarray(ts, float), idx.astype(float) + 1, meta


def session_spikes(sdir: Path, files, n_probes: int, grid: np.ndarray) -> dict:
    """Mean spikes per second per micro channel, on a common time grid."""
    per_channel, total = [], np.zeros(len(grid) - 1)
    n_ch = 0
    for e in files:
        if not e.name.lower().endswith((".nse", ".ntt")):
            continue
        got = spike_curve(Path(e.path), n_probes)
        if got is None:
            continue
        t, cum, meta = got
        stem, _ = V1.split_of(e.name.rsplit(".", 1)[0])
        n_ch += 1
        per_channel.append({
            "channel": stem, "file": e.name, "n_spikes": int(cum[-1]),
            "fs": meta["fs"], "threshold": meta["threshold"],
            "first": V1._nlx_dt(int(t[0] * 1e6)), "last": V1._nlx_dt(int(t[-1] * 1e6)),
            "rate_hz": round(float(cum[-1] / max(t[-1] - t[0], 1e-9)), 3),
        })
        # interpolate the cumulative curve onto the grid; outside the file's own span it
        # is flat, so a channel only contributes to the bins it actually covers
        c = np.interp(grid, t, cum, left=0.0, right=float(cum[-1]))
        total += np.diff(c)

    width = np.diff(grid)
    rate = np.divide(total, width * max(n_ch, 1), out=np.zeros_like(total),
                     where=width > 0)
    rates = sorted({c["fs"] for c in per_channel if c["fs"]})
    return {"n_files": len(per_channel), "n_channels": len({c["channel"] for c in per_channel}),
            "n_spikes": int(sum(c["n_spikes"] for c in per_channel)),
            "fs_hz": ", ".join(f"{int(f)}" for f in rates),
            "mean_rate_hz": round(float(np.mean([c["rate_hz"] for c in per_channel])), 2)
                            if per_channel else 0.0,
            "channels": per_channel, "rate": [float(x) for x in rate]}


# -------------------------------------------------------------------- signal windows


def band_power(P: np.ndarray, fr: np.ndarray, lo: float, hi: float) -> float:
    return float(P[(fr >= lo) & (fr < hi)].sum())


def comb_power(P, fr, f0: float, hi: float, avoid: float | None = None) -> float:
    """Power in a harmonic comb, skipping harmonics that sit on top of another comb."""
    out, k = 0.0, 1
    while f0 * k < hi:
        f = f0 * k
        if avoid is None or min(abs(f - avoid * j) for j in range(1, int(hi / avoid) + 2)) > 1.0:
            out += band_power(P, fr, f - 1.0, f + 1.0)
        k += 1
    return out


def window_metrics(x: np.ndarray, fs: float, adbitvolts: float, admax: int) -> dict:
    """One window of raw int16 -> the numbers the label is made from."""
    uv = x.astype(np.float64) * adbitvolts * 1e6
    rms = float(uv.std())
    sat = float(np.mean(np.abs(x) >= 0.995 * admax))
    distinct = int(min(len(np.unique(x[:2000])), 9999))

    w = np.hanning(len(uv))
    P = np.abs(np.fft.rfft((uv - uv.mean()) * w)) ** 2
    fr = np.fft.rfftfreq(len(uv), 1.0 / fs)
    broad = band_power(P, fr, *BROAD) or 1e-30
    nyq = min(BROAD[1], fs / 2)
    mains = comb_power(P, fr, MAINS_HZ, nyq)
    rail = comb_power(P, fr, RAIL_HZ, nyq, avoid=MAINS_HZ)
    lf = band_power(P, fr, 1, 8)
    hf = band_power(P, fr, 60, 120) or 1e-30

    m = {"rms_uv": round(rms, 2), "sat_frac": round(sat, 5),
         "mains_ratio": round(mains / broad, 3), "rail_ratio": round(rail / broad, 3),
         "lf_hf": round(lf / hf, 2), "distinct": distinct}
    m["label"] = label_window(m)
    return m


def label_window(m: dict) -> str:
    """See LABEL_RULES.

    Saturation is tested FIRST. A channel pinned against the rail has no variance at
    all, so every "is it flat" test calls it empty - and a railed amplifier and an
    unplugged one need different things done about them.
    """
    if m["sat_frac"] > 0.01:
        return "saturated"
    if m["rms_uv"] < 1.0 or m["distinct"] < 3:
        return "empty"
    if m["mains_ratio"] > 0.5:
        return "mains"
    if m["rail_ratio"] > 0.2:
        return "railway"
    if m["lf_hf"] >= 3.0 and 5.0 <= m["rms_uv"] <= 500.0:
        return "LFP-like"
    return "unclear"


def probe_file(path: Path, size: int, every_s: float, max_probes: int,
               win_s: float) -> list[dict]:
    """Sample windows evenly through one .ncs and measure each."""
    n_rec = (size - HDR_BYTES) // REC_BYTES
    if n_rec < 4:
        return []
    with path.open("rb") as fh:
        h = V1.read_header(fh)
        try:
            fs = float(h.get("SamplingFrequency", ""))
            adbv = float(h.get("ADBitVolts", ""))
            admax = int(float(h.get("ADMaxValue", "32767")))
        except ValueError:
            return []
        if fs <= 0:
            return []

        win_rec = max(2, int(round(win_s * fs / SAMP_PER_REC)))
        recorded_s = n_rec * SAMP_PER_REC / fs
        k = int(np.clip(recorded_s / max(every_s, 1.0), 1, max_probes))
        starts = np.unique(np.linspace(0, max(n_rec - win_rec, 0), k).astype(int))

        out = []
        for i in starts:
            fh.seek(HDR_BYTES + int(i) * REC_BYTES)
            buf = fh.read(win_rec * REC_BYTES)
            if len(buf) < win_rec * REC_BYTES:
                continue
            ts = struct.unpack("<Q", buf[:8])[0]
            x = np.concatenate([
                np.frombuffer(buf[j * REC_BYTES + 20:(j + 1) * REC_BYTES], "<i2")
                for j in range(win_rec)
            ])
            m = window_metrics(x, fs, adbv, admax)
            m["t"] = V1._nlx_dt(ts)
            m["file"] = path.name
            out.append(m)
    return out


def session_probes(sdir: Path, files, n_channels: int, every_s: float,
                   max_probes: int, win_s: float) -> list[dict]:
    """Probe a few macro channels, spread across leads, through every block."""
    by_key: dict[tuple[str, int], tuple[int, str]] = {}
    for e in files:
        if not e.name.lower().endswith(".ncs"):
            continue
        stem, sp = V1.split_of(e.name[:-4])
        fam = V1.family_of(stem)
        if V1.stream_kind([fam]) == "micro":       # spikes cover the micro side
            continue
        try:
            size = e.stat().st_size
        except OSError:
            continue
        key = (fam, sp)
        if key not in by_key or size > by_key[key][0]:
            by_key[key] = (size, e.name)

    fams = sorted({fam for fam, _ in by_key})
    keep = set(fams[:: max(1, len(fams) // max(n_channels, 1))][:n_channels] or fams[:n_channels])
    out = []
    for (fam, sp), (size, name) in sorted(by_key.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if fam not in keep:
            continue
        try:
            for m in probe_file(sdir / name, size, every_s, max_probes, win_s):
                m["family"] = fam
                m["split"] = sp
                out.append(m)
        except OSError as err:
            print(f"    ! cannot probe {name} - {err}")
    out.sort(key=lambda m: m["t"])
    return out


# ------------------------------------------------------------------------- scanning


def scan_session(folder: Path, sdir: Path, files, args) -> dict:
    rel = sdir.relative_to(folder)
    dur = V1.scan_session(folder, sdir, files, args.gap_min)

    mk = session_markers(sdir, args.burst_gap)
    start, end = dur["start"], dur["end"]

    if start and end and end > start:
        grid = np.linspace(start.timestamp(), end.timestamp(), args.spike_bins + 1)
    else:
        grid = np.linspace(0.0, 1.0, args.spike_bins + 1)
    sp = session_spikes(sdir, files, args.spike_probes, grid)

    probes = ([] if args.no_signal else
              session_probes(sdir, files, args.probe_channels, args.probe_every * 60,
                             args.max_probes, args.window))

    counts = Counter(p["label"] for p in probes)
    fired = {n: t["n"] for n, t in mk["trains"].items() if t["n"]}
    silent = sorted(n for n, t in mk["trains"].items() if not t["n"])

    print(f"  {rel.as_posix():<46} {fmt_hms(dur['recorded_s']):>9} recorded")
    if fired:
        for name, t in sorted(mk["trains"].items()):
            if t["n"]:
                print(f"      marker {name:<12} {t['n']:>6}x  {t['verdict']}")
    if silent:
        print(f"      declared but never fired: {', '.join(silent)}")
    if sp["n_spikes"]:
        print(f"      spikes {sp['n_spikes']:>11,} on {sp['n_channels']} channels"
              f" at {sp['fs_hz']} Hz, {sp['mean_rate_hz']}/s per channel")
    if probes:
        print("      signal " + "  ".join(f"{k} {100 * v / len(probes):.0f}%"
                                          for k, v in counts.most_common()))

    return {
        "folder": folder.name, "patient": dur["patient"], "session": rel.as_posix(),
        "label": rel.as_posix(), "path": str(sdir),
        "start": start, "end": end, "acq_created": dur["acq_created"],
        "acq_closed": dur["acq_closed"], "recorded_s": dur["recorded_s"],
        "streams": dur["streams"], "notes": dur["notes"],
        "markers": mk, "spikes": sp, "spike_grid": [float(g) for g in grid],
        "probes": probes, "probe_counts": dict(counts),
    }


def scan_folder(folder: Path, args) -> list[dict]:
    print(f"\n{folder.name}")
    out = [scan_session(folder, sdir, files, args)
           for sdir, files in V1.walk_sessions(folder)]
    if not out:
        print("  (no .ncs anywhere in this folder)")
    return sorted(out, key=lambda s: (s["start"] or datetime.max, s["label"]))


# ----------------------------------------------------------------------------- plot

PROBE_COLOUR = {
    "LFP-like": "#2ca02c", "mains": "#d62728", "railway": "#ff7f0e",
    "saturated": "#9467bd", "empty": "#7f7f7f", "unclear": "#c7c7c7",
}
LANE_ORDER = ["LFP-like", "railway", "mains", "saturated", "empty", "unclear"]


def plot_folder(folder_name: str, sessions: list[dict], src: Path, path: Path,
                pause_s: float):
    """v1's timeline, with a signal lane, a marker lane and a spike lane beneath it."""
    plotted = [s for s in sessions if s["start"]]
    if not plotted:
        print(f"  {folder_name}: nothing datable to plot")
        return

    n = len(plotted)
    multi_patient = len({s["patient"] for s in plotted}) > 1
    fig, (axT, axB) = plt.subplots(
        1, 2,
        figsize=(15.0, max(4.0, 1.9 + 1.15 * n)),
        gridspec_kw=dict(width_ratios=[3.0, 1.15], wspace=0.04),
        sharey=True,
    )

    # lanes inside one session row, top to bottom
    LANES = ("rec", "signal", "markers", "spikes")
    H = 0.80
    lane_h = H / len(LANES)
    ylab, span, acq_ends = [], [], []

    def lane_y(y, i):
        return y + H / 2 - lane_h * (i + 0.5)

    for i, s in enumerate(plotted):
        y = n - 1 - i
        day = s["start"]
        hid = lambda t: V1.hours_into_day(t, day)  # noqa: E731 - local shorthand

        has_data = any(g["start"] for st in s["streams"] for g in st["segments"])
        if s["acq_created"] and s["acq_closed"]:
            a0, a1 = hid(s["acq_created"]), hid(s["acq_closed"])
            axT.barh(y, a1 - a0, left=a0, height=H, color="0.5", alpha=0.08,
                     edgecolor="0.7", linewidth=0.5, zorder=1)
            acq_ends.append(a1)
            if not has_data:
                span += [a0, a1]

        # --- lane 1: recording, one thin bar per stream, cut at every pause
        streams = s["streams"] or []
        sub = lane_h / max(len(streams), 1)
        for j, st in enumerate(streams):
            yy = lane_y(y, 0) + lane_h / 2 - sub * (j + 0.5)
            c = V1.STREAM_COLOUR.get(st["kind"], V1.STREAM_COLOUR[""])
            for seg in st["segments"]:
                if not seg["start"]:
                    continue
                cuts = [p for p in st["pauses"]
                        if seg["start"] <= p["start"] and p["end"] <= seg["end"]]
                on = seg["start"]
                for p in cuts + [{"start": seg["end"], "end": seg["end"]}]:
                    x0 = hid(on)
                    w = max((p["start"] - on).total_seconds() / 3600, 0.004)
                    axT.barh(yy, w, left=x0, height=sub * 0.8, color=c, zorder=3)
                    span += [x0, x0 + w]
                    on = p["end"]

        # --- lane 2: signal probes, one tick each, coloured by what the window looks like
        for p in s["probes"]:
            axT.barh(lane_y(y, 1), 0.055, left=hid(p["t"]) - 0.027,
                     height=lane_h * 0.72, color=PROBE_COLOUR.get(p["label"], "#c7c7c7"),
                     zorder=3)

        # --- lane 3: marker bursts, and the reference changes
        for name, tr in sorted(s["markers"]["trains"].items()):
            for b in tr["bursts"]:
                x0 = hid(V1._nlx_dt(int(b["start"] * 1e6)))
                x1 = hid(V1._nlx_dt(int(b["end"] * 1e6)))
                axT.barh(lane_y(y, 2), max(x1 - x0, 0.02), left=x0,
                         height=lane_h * 0.72, color="#1a1a1a", zorder=3)
        for rc in s["markers"]["ref_changes"]:
            axT.plot([hid(V1._nlx_dt(int(rc["t"] * 1e6)))], [lane_y(y, 2)],
                     marker="v", ms=4, color="#8c564b", zorder=4)

        # --- lane 4: spike rate, as a strip whose darkness is the rate
        grid = np.asarray(s["spike_grid"], float)
        rate = np.asarray(s["spikes"]["rate"], float)
        if len(grid) > 1 and rate.size and rate.max() > 0:
            hours = [(V1._nlx_dt(int(g * 1e6)) - day.replace(hour=0, minute=0, second=0,
                                                             microsecond=0)).total_seconds() / 3600
                     for g in grid]
            norm = rate / rate.max()
            for a, b, v in zip(hours[:-1], hours[1:], norm):
                if v <= 0:
                    continue
                axT.barh(lane_y(y, 3), b - a, left=a, height=lane_h * 0.72,
                         color=plt.get_cmap("viridis")(0.15 + 0.85 * v), zorder=3)

        tag = f"{s['patient']}  " if multi_patient else ""
        ylab.append(f"{day:%Y-%m-%d}  {tag}{short(s['label'], 30)}")

        # --- right panel: what the probes said, as one stacked bar
        tot = sum(s["probe_counts"].values())
        left = 0.0
        for lab in LANE_ORDER:
            v = s["probe_counts"].get(lab, 0)
            if not v:
                continue
            axB.barh(lane_y(y, 0), 100 * v / tot, left=left, height=lane_h * 1.6,
                     color=PROBE_COLOUR[lab], zorder=3)
            left += 100 * v / tot
        bits = []
        if tot:
            bits.append(f"{tot} probes")
        fired = {nm: tr for nm, tr in s["markers"]["trains"].items() if tr["n"]}
        bits.append(", ".join(f"{nm} x{tr['n']}" for nm, tr in sorted(fired.items()))
                    or "no markers")
        if s["spikes"]["n_spikes"]:
            bits.append(f"{s['spikes']['n_spikes']:,} spikes / {s['spikes']['n_channels']} ch")
        axB.text(1.0, lane_y(y, 2), "\n".join(bits), va="center", ha="left",
                 fontsize=6.8, color="0.25")

    axT.set_yticks(range(n))
    axT.set_yticklabels(ylab[::-1], fontsize=7.4, family="monospace")
    axT.set_ylim(-0.75, n - 0.25)
    # name the lanes on the axis itself - four unlabelled strips per row are unreadable
    # from a legend alone, and the reader has to know which strip is which to use them
    axT.set_yticks([lane_y(y, k) for y in range(n) for k in range(len(LANES))],
                   minor=True)
    axT.set_yticklabels(list(LANES) * n, minor=True, fontsize=5.4, color="0.55")
    axT.tick_params(axis="y", which="minor", length=0, pad=1)
    for i, s in enumerate(plotted):          # a faint rule between sessions
        if i:
            axT.axhline(n - 0.5 - i, color="0.9", linewidth=0.7, zorder=0)

    lo = max(0.0, np.floor(min(span, default=0.0)) - 0.5)
    hi = np.ceil(max(span, default=24.0)) + 0.5
    clipped = any(a > hi for a in acq_ends)
    axT.set_xlim(lo, hi)
    step, ticks = V1.tick_ladder(lo, hi)
    axT.set_xticks(ticks)
    axT.set_xticklabels(
        [("" if step % 24 == 0 else f"{int(t) % 24:d}h")
         + (f"\n+{int(t) // 24}d" if t >= 24 else "") for t in ticks], fontsize=8)
    for boundary in np.arange(24, hi, 24):
        axT.axvline(boundary, color="0.75", linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    axT.set_xlim(lo, hi)
    axT.set_xlabel("time of day, from the start of each session's own day", fontsize=9)
    axT.grid(axis="x", color="0.92", linewidth=0.6, zorder=0)
    axT.set_axisbelow(True)

    axB.set_xlim(0, 100)
    axB.set_xticks([0, 50, 100])
    axB.set_xticklabels(["0", "50", "100%"], fontsize=8)
    axB.set_xlabel("share of signal probes", fontsize=9)
    axB.grid(axis="x", color="0.92", linewidth=0.6, zorder=0)
    axB.set_axisbelow(True)
    for ax in (axT, axB):
        for sp_ in ("top", "right", "left"):
            ax.spines[sp_].set_visible(False)
        ax.tick_params(axis="y", length=0)

    lanes_note = "rows, top to bottom:  recording  |  signal  |  markers  |  spike rate"
    axT.set_title(f"{folder_name}  -  what was recorded    ({lanes_note})",
                  fontsize=11, loc="left", pad=30)

    keys = [k for k in LANE_ORDER if any(k in s["probe_counts"] for s in plotted)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=PROBE_COLOUR[k]) for k in keys]
    labels = list(keys)
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#1a1a1a"))
    labels.append("marker burst")
    handles.append(plt.Line2D([], [], marker="v", ls="", color="#8c564b"))
    labels.append("reference changed")
    axT.legend(handles, labels, fontsize=8, ncol=min(len(labels), 8), frameon=False,
               loc="lower left", bbox_to_anchor=(0, 1.005))

    fig.text(
        0.005, -0.55 / fig.get_figheight(),
        f"signal labels are 2 s windows on macro channels, thresholds in LABEL_RULES; "
        f"spike lane is viridis-scaled per session"
        f"{'; grey band clipped where it runs past the data' if clipped else ''}  |  "
        f"{src}  |  {Path(__file__).name}  {datetime.now():%Y-%m-%d %H:%M}",
        fontsize=6.5, color="0.55", ha="left", va="top")

    V1.save_png(fig, path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


# ------------------------------------------------------------------------------ csv

MARKER_COLS = ["folder", "patient", "session", "name", "port", "bit", "declared",
               "fired", "n_pulses", "first", "last", "median_isi_s", "regular_frac",
               "n_bursts", "verdict"]
PROBE_COLS = ["folder", "patient", "session", "t", "family", "split", "file", "label",
              "rms_uv", "sat_frac", "mains_ratio", "rail_ratio", "lf_hf", "distinct"]
SPIKE_COLS = ["folder", "patient", "session", "channel", "file", "fs", "threshold",
              "n_spikes", "first", "last", "rate_hz"]


def marker_rows(sessions):
    out = []
    for s in sessions:
        where = {d["name"]: d for d in s["markers"]["declared"]}
        for name, tr in sorted(s["markers"]["trains"].items()):
            d = where.get(name, {})
            b = tr["bursts"]
            out.append({
                "folder": s["folder"], "patient": s["patient"], "session": s["session"],
                "name": name, "port": d.get("port", ""), "bit": d.get("bit", ""),
                "declared": "yes" if name in where else "",
                "fired": "yes" if tr["n"] else "no", "n_pulses": tr["n"],
                "first": f"{V1._nlx_dt(int(b[0]['start'] * 1e6)):%Y-%m-%d %H:%M:%S}" if b else "",
                "last": f"{V1._nlx_dt(int(b[-1]['end'] * 1e6)):%Y-%m-%d %H:%M:%S}" if b else "",
                "median_isi_s": round(tr["median_isi"], 4) if tr["median_isi"] else "",
                "regular_frac": round(tr["regular_frac"], 3),
                "n_bursts": len(b), "verdict": tr["verdict"],
            })
    return out


def probe_rows(sessions):
    return [
        {"folder": s["folder"], "patient": s["patient"], "session": s["session"],
         "t": f"{p['t']:%Y-%m-%d %H:%M:%S}", "family": p.get("family", ""),
         "split": p.get("split", ""), "file": p["file"], "label": p["label"],
         "rms_uv": p["rms_uv"], "sat_frac": p["sat_frac"], "mains_ratio": p["mains_ratio"],
         "rail_ratio": p["rail_ratio"], "lf_hf": p["lf_hf"], "distinct": p["distinct"]}
        for s in sessions for p in s["probes"]
    ]


def spike_rows(sessions):
    return [
        {"folder": s["folder"], "patient": s["patient"], "session": s["session"],
         "channel": c["channel"], "file": c["file"],
         "fs": int(c["fs"]) if c["fs"] else "", "threshold": c["threshold"],
         "n_spikes": c["n_spikes"],
         "first": f"{c['first']:%Y-%m-%d %H:%M:%S}", "last": f"{c['last']:%Y-%m-%d %H:%M:%S}",
         "rate_hz": c["rate_hz"]}
        for s in sessions for c in s["spikes"]["channels"]
    ]


def report(sessions, pause_s):
    """The photodiode question, answered per patient, plus what the signal looked like."""
    by = defaultdict(list)
    for s in sessions:
        by[(s["patient"], s["folder"])].append(s)
    for (patient, folder), ss in sorted(by.items()):
        print("_" * 96)
        print(f"\n  {patient} in {folder} - WHAT WAS RECORDED\n")
        decl = {d["name"]: d for s in ss for d in s["markers"]["declared"]}
        fired = Counter()
        for s in ss:
            for name, tr in s["markers"]["trains"].items():
                fired[name] += tr["n"]
        print("    Marker lines")
        for name in sorted(set(decl) | set(fired)):
            d = decl.get(name, {})
            at = f"port {d['port']} bit {d['bit']}" if d else "not in the log"
            if fired[name]:
                v = next((s["markers"]["trains"][name]["verdict"] for s in ss
                          if s["markers"]["trains"].get(name, {}).get("n")), "")
                print(f"      {name:<14} {at:<16} {fired[name]:>7} pulses   {v}")
            else:
                print(f"      {name:<14} {at:<16} {'DECLARED, NEVER FIRED':>7}")
        sp = sum(s["spikes"]["n_spikes"] for s in ss)
        if sp:
            rates = sorted({s["spikes"]["fs_hz"] for s in ss if s["spikes"]["fs_hz"]})
            per = [c["rate_hz"] for s in ss for c in s["spikes"]["channels"]]
            print(f"\n    Threshold crossings   : {sp:,} on "
                  f"{max((s['spikes']['n_channels'] for s in ss), default=0)}"
                  f" spike channels at {'/'.join(rates)} Hz")
            print(f"      per channel         : median {np.median(per):.1f} /s"
                  + ("   - loose enough to be catching noise, not units"
                     if np.median(per) > 5 else ""))
        c = Counter()
        for s in ss:
            c.update(s["probe_counts"])
        tot = sum(c.values())
        if tot:
            print(f"    Signal probes ({tot})  : "
                  + ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in c.most_common()))
        rc = [r for s in ss for r in s["markers"]["ref_changes"]]
        if rc:
            print(f"    Reference changes     : {len(rc)}"
                  f" (every later sample means something else)")
            for r in sorted(rc, key=lambda x: x["t"])[:8]:
                print(f"      {V1._nlx_dt(int(r['t'] * 1e6)):%Y-%m-%d %H:%M:%S}"
                      f"  {r['transition']}  on {r['n_channels']} channels")
    print("_" * 96)


# ------------------------------------------------------------------------------ main


def _enc(o):
    return o.isoformat() if isinstance(o, datetime) else o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=V1.OUT)
    ap.add_argument("--folder", action="append", default=None)
    ap.add_argument("--no-signal", action="store_true",
                    help="skip the sample reads - markers and spikes only")
    ap.add_argument("--probe-channels", type=int, default=3,
                    help="macro channels probed per session (default 3)")
    ap.add_argument("--probe-every", type=float, default=15.0, metavar="MIN",
                    help="one signal window per this many minutes of block (default 15)")
    ap.add_argument("--max-probes", type=int, default=40,
                    help="cap on windows per channel per block (default 40)")
    ap.add_argument("--window", type=float, default=2.0, metavar="SEC",
                    help="length of each signal window (default 2 s)")
    ap.add_argument("--spike-probes", type=int, default=64,
                    help="timestamps read per .nse to build its rate curve (default 64)")
    ap.add_argument("--spike-bins", type=int, default=240)
    ap.add_argument("--burst-gap", type=float, default=5.0, metavar="SEC",
                    help="a marker gap this long starts a new burst (default 5)")
    ap.add_argument("--gap-min", type=float, default=V1.GAP_MIN_S)
    ap.add_argument("--pause-threshold", type=float, default=V1.PAUSE_S / 60, metavar="MIN")
    ap.add_argument("--use-cache", action="store_true")
    args = ap.parse_args(argv)
    args.pause_threshold *= 60
    args.root = resolve = V1.resolve_data_root(args.root)
    print(f"MICROEPI: {resolve}")
    print(f"labels:{LABEL_RULES}")

    folders = sorted(p for p in args.root.glob(V1.FOLDER_GLOB) if p.is_dir())
    if args.folder:
        folders = [f for f in folders if any(k.lower() in f.name.lower() for k in args.folder)]
    if not folders:
        print(f"no {V1.FOLDER_GLOB} folders under {args.root}")
        return 1

    cache = args.outdir / "_activity_cache.json"
    scanned: dict[str, list[dict]] = {}
    if args.use_cache:
        if not cache.exists():
            print(f"no cache at {cache} - run without --use-cache first")
            return 1
        raw = json.loads(cache.read_text(encoding="utf-8"))
        scanned = {k: [_dec(s) for s in v] for k, v in raw.items()}
        print(f"redrawing from {cache}")
    else:
        for f in folders:
            scanned[f.name] = scan_folder(f, args)
        V1.save_bytes(json.dumps(scanned, default=_enc, indent=1).encode("utf-8"), cache)

    print()
    every = []
    for f in folders:
        ss = scanned.get(f.name, [])
        every += ss
        if not ss:
            continue
        plot_folder(f.name, ss, args.root / f.name,
                    args.outdir / f"{f.name}_activity.png", args.pause_threshold)
        V1.save_csv(marker_rows(ss), MARKER_COLS, args.outdir / f"{f.name}_markers.csv")
        V1.save_csv(spike_rows(ss), SPIKE_COLS, args.outdir / f"{f.name}_spikes.csv")
        if not args.no_signal:
            V1.save_csv(probe_rows(ss), PROBE_COLS, args.outdir / f"{f.name}_probes.csv")

    if every:
        V1.save_csv(marker_rows(every), MARKER_COLS, args.outdir / "microepi_nlx_markers.csv")
        V1.save_csv(spike_rows(every), SPIKE_COLS, args.outdir / "microepi_nlx_spikes.csv")
        if not args.no_signal:
            V1.save_csv(probe_rows(every), PROBE_COLS, args.outdir / "microepi_nlx_probes.csv")
    report(every, args.pause_threshold)
    print(f"\n  -> {args.outdir}")
    return 0


DT_KEYS = {"start", "end", "acq_created", "acq_closed", "t", "first", "last"}


def _dec(o):
    if isinstance(o, dict):
        return {k: _dec_val(k, v) for k, v in o.items()}
    if isinstance(o, list):
        return [_dec(x) for x in o]
    return o


def _dec_val(k, v):
    if isinstance(v, str) and k in DT_KEYS:
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            return v
    return _dec(v) if isinstance(v, (dict, list)) else v


if __name__ == "__main__":
    sys.exit(main())
