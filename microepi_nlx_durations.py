#!/usr/bin/env python3
"""
microepi_nlx_durations.py - how much was actually recorded in each MicroEPI-B-* folder?

The question this answers: for every Neuralynx (Pegasus / ATLAS) session sitting under
DATARAW/MICROEPI/MicroEPI-B-##, when did it run and how many minutes of signal does it
really contain? "Recording" is not the same as "acquisition open": Pegasus keeps a file
open across pauses, so a session whose header says four hours can hold twenty minutes of
samples. Both numbers are reported, and the difference is drawn.

How the durations are read
  A .ncs file is a 16384-byte ASCII header followed by 1044-byte records
  (uint64 timestamp us, uint32 channel, uint32 sampling rate, uint32 valid samples,
  512 int16). Reading the FIRST and the LAST record of a file gives:

      span      = (ts_last - ts_first) / 1e6 + n_valid_last / fs   wall time covered
      recorded  = ((n_records - 1) * 512 + n_valid_last) / fs      signal actually stored
      gap       = span - recorded                                  paused / dropped time

  so one 16 KB header read plus two 20-byte seeks per file, instead of reading gigabytes.
  All channels of one split share the same clock, so only ONE representative file per
  (channel family, split index) is opened - e.g. one csclead_* and one lead_* per split.
  Records other than the last are assumed full (512 samples), which is how Pegasus writes
  them; a file whose size is not a whole number of records is flagged as truncated.

  Record timestamps are microseconds that decode to the acquisition wall clock
  (verified: EL043 csclead_A_L-101 starts at 11:38:56, and the paired export is named
  EL043_20250617_11h38m56_226min.h5), so they are read with a UTC conversion and used
  as local time - no timezone is applied.

Micro and macro are two recordings, not one
  A session folder holds both streams - micro_* / csclead_* at 32 kHz and lead_* at
  2 kHz - and they need not run together. In EL043 the macro stops after split 9 while
  the micro runs on to split 15, six and a half hours further. So every duration here
  is per stream; the one session-level number is the LONGEST stream, never a sum and
  never a per-split blend of the two, which would be a duration neither stream has.

Outputs, one set per MicroEPI-B-## folder, under outputs/microepi_nlx/
  <folder>_durations.png   time-of-day timeline + recorded-duration bars, per stream
  <folder>_sessions.csv    one row per session
  <folder>_streams.csv     one row per stream (this is the table of durations)
  <folder>_segments.csv    one row per stream x split (Pegasus stop/start segment)
  <folder>_pauses.csv      every stop, with the ones over --pause-threshold marked
  microepi_nlx_{sessions,streams,segments,pauses}.csv   all folders together
  _scan_cache.json         raw scan, so replotting does not re-read the share

Where the recording stopped
  Coverage, and the table of prolonged stops, follow the MicroEPI-G Blackrock report.
  The stops themselves are found by BISECTING the record timestamps (see find_pauses),
  which locates every gap to the record without reading the samples - the alternative,
  reading EL049 end to end, is 384 GB across the share.

Where the data is
    DATARAW/MICROEPI is found relative to THIS file - it sits in ANALYSIS/FLM/
    Analysis_LoraFanda, and DATARAW is a sibling of ANALYSIS - so the script works
    whether the tree is reached as S:\\HumanNeuronLab (that is \\\\isis.unige.ch\\medecine)
    or as \\\\nasac-m2.unige.ch\\m-HumanNeuronLab, which is the same folders with no
    HumanNeuronLab level inside it. Override with --root or $MICROEPI_ROOT.

Usage
    python microepi_nlx_durations.py                 # scan + plot everything
    python microepi_nlx_durations.py --dry-run       # just list the sessions found
    python microepi_nlx_durations.py --folder B-02   # one folder
    python microepi_nlx_durations.py --use-cache     # replot from the last scan
    python microepi_nlx_durations.py --root D:\\MICROEPI      # a copy somewhere else
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "microepi_nlx"
FOLDER_GLOB = "MicroEPI-B-*"

# A pause shorter than this is not looked for (it would not be visible on the figure
# either); a pause at least PAUSE_S long is called out as one worth explaining, the
# same 15 minutes the Blackrock/MicroEPI-G report uses.
GAP_MIN_S = 60.0
PAUSE_S = 15 * 60.0

HDR_BYTES = 16384
REC_BYTES = 1044
SAMP_PER_REC = 512

# Directories never descended into. The first four are Pegasus / macOS / Windows
# bookkeeping; the rest are the unrelated trees on the Seagate backup copied into
# MicroEPI-B-03, which hold tens of thousands of files and no electrophysiology.
SKIP_DIRS = {
    "ConfigurationLog",
    ".fseventsd",
    ".Trashes",
    "$RECYCLE.BIN",
    "System Volume Information",
    "ITS Backup",
    "Lisa - phone pictures",
    "Syndyse Files",
    "Seagate",
}

PATIENT_RE = re.compile(r"^(EL\d+|PAT_\d+|P\d{2,})$", re.I)
SPLIT_RE = re.compile(r"^(?P<stem>.+)_(?P<split>\d{4})$")
FAMILY_RE = re.compile(r"^(?P<fam>.+?)-\d+$")


# ------------------------------------------------------------------------ where MICROEPI is


def _derived_root() -> Path | None:
    """DATARAW/MICROEPI as a sibling of ANALYSIS, from wherever THIS file sits.

    ANALYSIS/FLM/Analysis_LoraFanda is three levels under the lab root, and DATARAW is
    next to ANALYSIS, so this is correct on every mount without naming any of them.
    """
    try:
        return ROOT.parents[2] / "DATARAW" / "MICROEPI"
    except IndexError:  # script copied somewhere shallow
        return None


# The lab tree answers to more than one name - S: is \\isis.unige.ch\medecine (so the
# lab root is S:\HumanNeuronLab), while the same folders are the ROOT of the share
# \\nasac-m2.unige.ch\m-HumanNeuronLab, with no HumanNeuronLab level inside it. A path
# hard-coded for one mount is wrong on the other, so the derived one is tried first and
# these are only fallbacks for running the script from outside the repo.
DATA_ROOT_CANDIDATES = [
    p
    for p in (
        _derived_root(),
        Path(r"S:\HumanNeuronLab\DATARAW\MICROEPI"),
        Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\MICROEPI"),
        Path(r"\\isis.unige.ch\medecine\HumanNeuronLab\DATARAW\MICROEPI"),
    )
    if p is not None
]


def resolve_data_root(explicit: Path | None) -> Path:
    """--root wins, then $MICROEPI_ROOT, then the first candidate that is really there."""
    for chosen, why in ((explicit, "--root"), (os.environ.get("MICROEPI_ROOT"), "MICROEPI_ROOT")):
        if chosen:
            p = Path(chosen)
            if not p.is_dir():
                raise SystemExit(f"{why} {p} is not a directory")
            return p
    for c in DATA_ROOT_CANDIDATES:
        try:
            if c.is_dir():
                return c
        except OSError:
            continue
    raise SystemExit(
        "could not find the MICROEPI directory. Tried:\n  "
        + "\n  ".join(str(c) for c in DATA_ROOT_CANDIDATES)
        + "\nPass the right one with --root, or set MICROEPI_ROOT."
    )


# --------------------------------------------------------------------------- writing


def _verified_write(write_fn, path: Path, check_fn, tries: int = 6):
    """Write to a local temp file, CHECK it, then copy it in and check again.

    Writes to this share have truncated a finished file to zero or half its length and
    still reported success, and open(path, "w") has already destroyed the old file by
    the time that happens. Nothing here goes straight to S:.
    """
    tmp = Path(tempfile.gettempdir()) / f"_nlxdur_{path.name}"
    last = ""
    for i in range(tries):
        try:
            write_fn(tmp)
            check_fn(tmp)
            n = tmp.stat().st_size
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tmp, path)
            if path.stat().st_size != n:
                raise OSError(f"{path.stat().st_size} of {n} bytes landed")
            check_fn(path)
            tmp.unlink(missing_ok=True)
            return n
        except Exception as e:  # noqa: BLE001 - retry anything the share throws
            last = f"{type(e).__name__}: {e}"
            print(f"    write attempt {i + 1} for {path.name} failed - {last}")
            time.sleep(2)
    raise SystemExit(f"could not write {path}: {last}")


def save_png(fig, path: Path, **kw):
    def _check(p):
        from PIL import Image

        with Image.open(p) as im:
            im.load()  # forces a full decode

    return _verified_write(lambda p: fig.savefig(p, **kw), path, _check)


def save_bytes(data: bytes, path: Path):
    """Byte-exact. Text mode would translate newlines on the way out and back, so a
    file that landed perfectly could still compare unequal."""

    def _check(p):
        got = Path(p).read_bytes()
        if got != data:
            raise OSError(f"{len(got)} of {len(data)} bytes landed")

    return _verified_write(lambda p: Path(p).write_bytes(data), path, _check)


def save_csv(rows: list[dict], cols: list[str], path: Path):
    if not rows:
        return 0
    buf = []
    buf.append(",".join(cols))
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            v = "" if v is None else str(v)
            if any(ch in v for ch in ',"\n'):
                v = '"' + v.replace('"', '""') + '"'
            vals.append(v)
        buf.append(",".join(vals))
    return save_bytes(("\n".join(buf) + "\n").encode("utf-8"), path)


# ------------------------------------------------------------------------ ncs reading


def _nlx_dt(ts_us: int) -> datetime:
    """Neuralynx record timestamp -> naive acquisition wall clock (see module docstring)."""
    return datetime.fromtimestamp(ts_us / 1e6, tz=timezone.utc).replace(tzinfo=None)


def _hdr_time(s: str) -> datetime | None:
    s = (s or "").strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def read_header(fh) -> dict[str, str]:
    txt = fh.read(HDR_BYTES).split(b"\x00", 1)[0].decode("latin-1")
    out: dict[str, str] = {}
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("-"):
            k, _, v = line[1:].partition(" ")
            out[k] = v.strip().strip('"')
    return out


def probe_ncs(path: Path, size: int) -> dict:
    """First + last record of one .ncs: when it ran, how much signal it holds."""
    rec = {
        "file": path.name,
        "bytes": size,
        "n_records": 0,
        "fs": None,
        "start": None,
        "end": None,
        "recorded_s": 0.0,
        "span_s": 0.0,
        "created": None,
        "closed": None,
        "note": "",
    }
    if size < HDR_BYTES:
        rec["note"] = "shorter than a header"
        return rec

    n_rec, tail = divmod(size - HDR_BYTES, REC_BYTES)
    if tail:
        rec["note"] = f"{tail} B past the last whole record - truncated?"

    with path.open("rb") as fh:
        h = read_header(fh)
        rec["created"] = _hdr_time(h.get("TimeCreated", ""))
        rec["closed"] = _hdr_time(h.get("TimeClosed", ""))
        try:
            rec["fs"] = float(h.get("SamplingFrequency", ""))
        except ValueError:
            pass
        if n_rec:
            fh.seek(HDR_BYTES)
            ts0, _, fs0, _ = struct.unpack("<QIII", fh.read(20))
            fh.seek(HDR_BYTES + (n_rec - 1) * REC_BYTES)
            ts1, _, fs1, nv1 = struct.unpack("<QIII", fh.read(20))
            fs = float(fs1 or fs0 or rec["fs"] or 0)
            if fs <= 0:
                rec["note"] = (rec["note"] + "; no sampling rate").lstrip("; ")
                return rec
            rec["fs"] = fs
            rec["n_records"] = n_rec
            rec["start"] = _nlx_dt(ts0)
            rec["span_s"] = (ts1 - ts0) / 1e6 + nv1 / fs
            rec["end"] = rec["start"] + timedelta(seconds=rec["span_s"])
            rec["recorded_s"] = ((n_rec - 1) * SAMP_PER_REC + nv1) / fs
    return rec


def find_pauses(path: Path, n_rec: int, fs: float, gap_min_s: float) -> list[tuple]:
    """Every gap of at least gap_min_s inside one .ncs, and where it starts and ends.

    Found by BISECTION on the record timestamps, never by reading the file. Let
    cum(i) = ts[i] - ts[0] - i * stride: the cumulative time missing before record i.
    It only ever increases, so if cum is the same at both ends of a range there is no
    gap anywhere inside it and the whole range can be dropped. Each test is one 8-byte
    read, so a 200 MB split costs a few hundred bytes instead of 200 MB - which is the
    only reason this is affordable on a share where reading EL049 in full would be
    hundreds of gigabytes.

    Records other than the last of a file are assumed full, as Pegasus writes them. A
    short record would show up as a fraction of a second of phantom gap, far below any
    threshold worth reporting.
    """
    if n_rec < 2 or fs <= 0:
        return []
    stride = SAMP_PER_REC / fs * 1e6
    gap_min_us = gap_min_s * 1e6
    seen: dict[int, int] = {}
    budget = 20000  # a pathological file cannot turn this into a full read

    with path.open("rb") as fh:

        def ts(i: int) -> int:
            if i not in seen:
                fh.seek(HDR_BYTES + i * REC_BYTES)
                seen[i] = struct.unpack("<Q", fh.read(8))[0]
            return seen[i]

        t0 = ts(0)

        def cum(i: int) -> float:
            return ts(i) - t0 - i * stride

        out: list[tuple] = []
        stack = [(0, n_rec - 1)]
        while stack and len(seen) < budget:
            a, b = stack.pop()
            if b - a < 1 or cum(b) - cum(a) < gap_min_us:
                continue
            if b - a == 1:
                out.append((_nlx_dt(int(ts(a) + stride)), _nlx_dt(ts(b))))
                continue
            m = (a + b) // 2
            stack += [(a, m), (m, b)]

    return sorted(out)


# -------------------------------------------------------------------------- scanning


def split_of(stem_name: str) -> tuple[str, int]:
    """csclead_A_L-101_0002 -> (csclead_A_L-101, 2);  lead_A_L-1 -> (lead_A_L-1, 0)."""
    m = SPLIT_RE.match(stem_name)
    return (m["stem"], int(m["split"])) if m else (stem_name, 0)


def family_of(stem: str) -> str:
    """csclead_A_L-101 -> csclead_A_L. One clock per family per split."""
    m = FAMILY_RE.match(stem)
    return m["fam"] if m else stem


def stream_kind(families) -> str:
    """What the files call themselves: micro_* and csclead_* are micro, lead_* is macro.

    Taken from the names rather than from the sampling rate, because the rate is what we
    would be guessing from. Anything else is left unnamed and shown by its rate alone.
    """
    fams = list(families)
    if fams and all(f.lower().startswith(("micro", "csc")) for f in fams):
        return "micro"
    if fams and all(f.lower().startswith("lead") for f in fams):
        return "macro"
    return ""


def walk_sessions(top: Path):
    """Yield (dir, [DirEntry]) for every directory that directly holds a .ncs file.

    DirEntry keeps the size that the directory listing already returned, so sizing 2340
    files in a session costs no extra round trip to the share.
    """
    stack = [str(top)]
    while stack:
        d = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError as e:
            print(f"  ! cannot list {d} - {e}")
            continue
        files = [e for e in entries if e.is_file(follow_symlinks=False)]
        if any(e.name.lower().endswith(".ncs") for e in files):
            yield Path(d), files
            continue  # a session directory has no session directories inside it
        for e in entries:
            if e.is_dir(follow_symlinks=False) and e.name not in SKIP_DIRS and not e.name.startswith("."):
                stack.append(e.path)


def patient_of(rel: Path) -> str:
    for part in rel.parts:
        if PATIENT_RE.match(part):
            return part.upper()
        m = re.match(r"^(EL\d+|PAT_\d+)_", part, re.I)
        if m:
            return m.group(1).upper()
    return rel.parts[0] if rel.parts else "?"


def scan_session(folder: Path, sdir: Path, files, gap_min_s: float = GAP_MIN_S) -> dict:
    """One Pegasus session directory -> its segments and its totals."""
    rel = sdir.relative_to(folder)
    groups: dict[tuple[str, int], tuple[int, str]] = {}
    channels: set[str] = set()
    ncs_bytes = 0
    n_ncs = 0
    other = defaultdict(int)

    for e in files:
        name = e.name
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        try:
            size = e.stat().st_size
        except OSError:
            size = 0
        if ext != "ncs":
            if ext in {"nev", "nse", "ntt", "nvt", "nrd"}:
                other[ext] += 1
            continue
        n_ncs += 1
        ncs_bytes += size
        stem, sp = split_of(name[:-4])
        channels.add(stem)
        key = (family_of(stem), sp)
        if key not in groups or size > groups[key][0]:
            groups[key] = (size, name)

    # one representative per (family, split)
    probes: list[dict] = []
    for (fam, sp), (size, name) in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        try:
            p = probe_ncs(sdir / name, size)
        except OSError as err:
            print(f"  ! cannot read {name} - {err}")
            continue
        p["family"] = fam
        p["split"] = sp
        probes.append(p)

    # Split by sampling rate FIRST. The micro and the macro stream are two recordings
    # that happen to share a folder, and they do not have to run together: in EL043 the
    # macro stops after split 9 while the micro keeps going to split 15, six and a half
    # hours further. Reducing a session to one number blends them into a duration that
    # neither stream has, so each is carried and reported on its own.
    streams: list[dict] = []
    by_fs = defaultdict(list)
    for p in probes:
        by_fs[int(p["fs"]) if p["fs"] else 0].append(p)

    for fs in sorted(by_fs, reverse=True):  # micro first
        ps = by_fs[fs]
        fams = sorted({p["family"] for p in ps})
        per_split: dict[int, dict] = {}
        for p in ps:
            cur = per_split.get(p["split"])
            if cur is None or p["recorded_s"] > cur["recorded_s"]:
                per_split[p["split"]] = p
        segs = [per_split[k] for k in sorted(per_split)]
        st = [s["start"] for s in segs if s["start"]]
        en = [s["end"] for s in segs if s["end"]]

        # inside ONE stream the families are the same clock, so any spread is an anomaly
        spread = defaultdict(list)
        for p in ps:
            spread[p["split"]].append(p["recorded_s"])
        within = max((max(v) - min(v) for v in spread.values() if len(v) > 1), default=0.0)

        # where this stream stopped recording: the gaps inside each block, plus the
        # dead time between one block and the next
        pauses = []
        for g in segs:
            if g["n_records"] > 1 and g["span_s"] - g["recorded_s"] >= gap_min_s:
                pauses += [
                    {"start": a, "end": b, "where": "inside a block"}
                    for a, b in find_pauses(sdir / g["file"], g["n_records"], g["fs"], gap_min_s)
                ]
        for prev, nxt in zip(segs, segs[1:]):
            if prev["end"] and nxt["start"] and (nxt["start"] - prev["end"]).total_seconds() >= gap_min_s:
                pauses.append({"start": prev["end"], "end": nxt["start"],
                               "where": "between blocks"})
        pauses.sort(key=lambda p: p["start"])
        for p in pauses:
            p["seconds"] = (p["end"] - p["start"]).total_seconds()

        recorded = float(sum(s["recorded_s"] for s in segs))
        wall = (max(en) - min(st)).total_seconds() if (st and en) else 0.0

        streams.append({
            "fs": fs,
            "kind": stream_kind(fams),
            "label": f"{stream_kind(fams)} {fs / 1000:g} kHz".strip(),
            "families": ", ".join(fams),
            "n_channels": len({c for c in channels if family_of(c) in set(fams)}),
            "n_segments": len(segs),
            "recorded_s": recorded,
            "offline_s": max(wall - recorded, 0.0),
            "coverage_pct": round(100 * recorded / wall, 1) if wall > 0 else 0.0,
            "pauses": pauses,
            "start": min(st) if st else None,
            "end": max(en) if en else None,
            "within_stream_spread_s": within,
            "segments": [
                {
                    "split": s["split"], "file": s["file"], "family": s["family"],
                    "fs": s["fs"], "n_records": s["n_records"],
                    "start": s["start"], "end": s["end"],
                    "recorded_s": s["recorded_s"], "span_s": s["span_s"],
                    "gap_s": s["span_s"] - s["recorded_s"], "note": s["note"],
                }
                for s in segs
            ],
        })

    created = [p["created"] for p in probes if p["created"]]
    closed = [p["closed"] for p in probes if p["closed"]]
    started = [s["start"] for s in streams if s["start"]]
    ended = [s["end"] for s in streams if s["end"]]

    # the headline number is the LONGEST single stream - a real duration of a real
    # recording - never a sum or a per-split blend of the two
    recorded_s = max((s["recorded_s"] for s in streams), default=0.0)
    start = min(started) if started else (min(created) if created else None)
    end = max(ended) if ended else (max(closed) if closed else None)

    notes = sorted({p["note"] for p in probes if p["note"]})
    for s in streams:
        if s["within_stream_spread_s"] > 1.0:
            notes.append(f"{s['label']} families differ by {s['within_stream_spread_s']:.1f}s")
    live = [s for s in streams if s["recorded_s"] > 0]
    if len(live) > 1:
        gap = max(s["recorded_s"] for s in live) - min(s["recorded_s"] for s in live)
        if gap > 60:
            notes.append(
                " vs ".join(f"{s['label']} {fmt_hms(s['recorded_s'])}" for s in live)
            )

    return {
        "folder": folder.name,
        "session": rel.as_posix(),
        "label": rel.as_posix(),
        "patient": patient_of(rel),
        "path": str(sdir),
        "start": start,
        "end": end,
        "recorded_s": recorded_s,
        "span_s": (end - start).total_seconds() if (start and end) else 0.0,
        "acq_open_s": (max(closed) - min(created)).total_seconds() if (created and closed) else 0.0,
        "acq_created": min(created) if created else None,
        "acq_closed": max(closed) if closed else None,
        "n_segments": max((s["n_segments"] for s in streams), default=0),
        "n_channels": len(channels),
        "n_ncs_files": n_ncs,
        "families": ", ".join(sorted({p["family"] for p in probes})),
        "fs_hz": " ".join(f"{s['fs']}" for s in streams),
        "recorded_by_stream": "  ".join(
            f"{s['label']} {fmt_hms(s['recorded_s'])}" for s in streams
        ),
        "gb": round(ncs_bytes / 1024**3, 2),
        "n_nev": other.get("nev", 0),
        "n_spike_files": other.get("nse", 0) + other.get("ntt", 0),
        "notes": "; ".join(notes),
        "streams": streams,
    }


def scan_folder(folder: Path, gap_min_s: float = GAP_MIN_S,
                pause_s: float = PAUSE_S) -> list[dict]:
    print(f"\n{folder.name}")
    out = []
    for sdir, files in walk_sessions(folder):
        s = scan_session(folder, sdir, files, gap_min_s)
        out.append(s)
        print(f"  {s['label']:<48} {s['n_channels']:>4} ch")
        for st in s["streams"]:
            long_p = [p for p in st["pauses"] if p["seconds"] >= pause_s]
            print(
                f"      {st['label']:<14} {fmt_hms(st['recorded_s']):>9} recorded"
                f"  {st['coverage_pct']:>5.1f}% coverage"
                f"  in {st['n_segments']:>2} block{'s' if st['n_segments'] != 1 else ' '}"
                f"  {st['n_channels']:>4} ch"
                + (f"   {len(long_p)} pause{'s' if len(long_p) != 1 else ''}"
                   f" over {fmt_hm(pause_s)}" if long_p else "")
            )
        if s["notes"]:
            print(f"      ! {s['notes']}")
    if not out:
        print("  (no .ncs anywhere in this folder)")
    return sorted(out, key=lambda s: (s["start"] or datetime.max, s["label"]))


# ----------------------------------------------------------------------------- cache


def _enc(o):
    return o.isoformat() if isinstance(o, datetime) else o


def _dec(o):
    if isinstance(o, dict):
        return {k: _dec_val(k, v) for k, v in o.items()}
    return o


def _dec_val(k, v):
    if isinstance(v, str) and k in {"start", "end", "acq_created", "acq_closed"}:
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            return None
    if isinstance(v, list):
        return [_dec(x) for x in v]
    if isinstance(v, dict):
        return _dec(v)
    return v


# ---------------------------------------------------------------------------- format


def fmt_hms(seconds: float) -> str:
    s = int(round(seconds or 0))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def fmt_hm(seconds: float) -> str:
    s = int(round(seconds or 0))
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}"


def short(label: str, n: int = 34) -> str:
    """Keep the ends of a long relative path - the middle is the redundant part."""
    if len(label) <= n:
        return label
    keep = (n - 3) // 2
    return label[:keep] + "..." + label[-(n - 3 - keep):]


# ----------------------------------------------------------------------------- plots


def hours_into_day(t: datetime, day: datetime) -> float:
    return (t - day.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600


def tick_ladder(lo: float, hi: float) -> tuple[float, list[float]]:
    """Hour ticks that stay readable whether a folder covers four hours or four days.

    B-03 holds two sessions a YEAR apart, one of them running for three days, so an
    absolute date axis cannot serve both; the axis is hours from the start of each
    session's own day, and only the tick spacing has to adapt.
    """
    for step in (1, 2, 3, 4, 6, 12, 24, 48):
        if (hi - lo) / step <= 12:
            break
    ticks = np.arange(np.floor(lo / step) * step, hi + 0.01, step)
    # Only ticks INSIDE the view. set_xticks widens the axis to take in any tick that
    # falls outside it - it quietly undoes set_xlim - which put a stray 0h and three
    # hours of empty margin on the left of every multi-day folder.
    ticks = [float(t) for t in ticks if lo <= t <= hi]
    return step, ticks or list(np.linspace(lo, hi, 4))


STREAM_COLOUR = {"micro": "#1f77b4", "macro": "#e8834a", "": "#7f7f7f"}


def plot_folder(folder_name: str, sessions: list[dict], src: Path, path: Path,
                pause_s: float = PAUSE_S):
    """Left: when each session ran, over the clock. Right: how much it actually holds.

    Micro and macro are drawn as their own bars, never merged - see scan_session.
    """
    plotted = [s for s in sessions if s["start"]]
    if not plotted:
        print(f"  {folder_name}: nothing datable to plot")
        return

    n = len(plotted)
    multi_patient = len({s["patient"] for s in plotted}) > 1
    kinds: list[str] = []
    for s in plotted:
        for st in s["streams"]:
            if st["label"] not in kinds:
                kinds.append(st["label"])
    kinds.sort(key=lambda k: (0 if k.startswith("micro") else 1 if k.startswith("macro") else 2, k))

    fig, (axT, axB) = plt.subplots(
        1, 2,
        figsize=(14.5, max(3.4, 1.9 + 0.62 * n)),
        gridspec_kw=dict(width_ratios=[3.0, 1.35], wspace=0.04),
        sharey=True,
    )

    # The axis is set by the DATA, not by how long a file happened to stay open: EL049
    # holds 47 h of signal inside a window Pegasus kept open for 197 h, and letting that
    # window set the range squeezes every bar into the left eighth of the plot. Sessions
    # with no data at all are the exception - their open window is all they have.
    ylab, span, acq_ends = [], [], []
    for i, s in enumerate(plotted):
        y = n - 1 - i  # first session at the top
        day = s["start"]
        rows = s["streams"] or [None]
        h = 0.74 / max(len(rows), 1)  # one lane per stream inside the row

        has_data = any(g["start"] for st in s["streams"] for g in st["segments"])
        if s["acq_created"] and s["acq_closed"]:
            a0 = hours_into_day(s["acq_created"], day)
            a1 = hours_into_day(s["acq_closed"], day)
            axT.barh(y, a1 - a0, left=a0, height=0.74, color="0.5", alpha=0.10,
                     edgecolor="0.6", linewidth=0.5, zorder=1)
            acq_ends.append(a1)
            if not has_data:
                span += [a0, a1]

        for j, st in enumerate(rows):
            if st is None:
                continue
            yy = y + 0.74 / 2 - h * (j + 0.5)
            c = STREAM_COLOUR.get(st["kind"], STREAM_COLOUR[""])
            # A block is drawn as the stretches that actually hold samples, cut at every
            # pause - drawing the block start-to-end would paint an eight-hour bar over
            # a stop the report lists as eight hours of nothing.
            for seg in st["segments"]:
                if not seg["start"]:
                    continue
                cuts = [p for p in st["pauses"]
                        if seg["start"] <= p["start"] and p["end"] <= seg["end"]]
                on = seg["start"]
                for p in cuts + [{"start": seg["end"], "end": seg["end"]}]:
                    x0 = hours_into_day(on, day)
                    w = max((p["start"] - on).total_seconds() / 3600, 0.004)
                    axT.barh(yy, w, left=x0, height=h * 0.86, color=c, zorder=3)
                    span += [x0, x0 + w]
                    on = p["end"]

            for p in st["pauses"]:  # the stops worth explaining, marked where they are
                if p["seconds"] < pause_s:
                    continue
                axT.barh(yy, p["seconds"] / 3600, left=hours_into_day(p["start"], day),
                         height=h * 0.86, color="#d62728", alpha=0.30, zorder=2)

            rec_h = st["recorded_s"] / 3600
            axB.barh(yy, rec_h, height=h * 0.86, color=c, zorder=3)
            axB.text(rec_h + 0.04, yy,
                     f"{fmt_hms(st['recorded_s'])}  {st['coverage_pct']:.0f}%"
                     f"  {st['n_channels']} ch",
                     va="center", ha="left", fontsize=7.0, color="0.25")

        tag = f"{s['patient']}  " if multi_patient else ""
        ylab.append(f"{day:%Y-%m-%d}  {tag}{short(s['label'])}")

    axT.set_yticks(range(n))
    axT.set_yticklabels(ylab[::-1], fontsize=7.6, family="monospace")
    axT.set_ylim(-0.7, n - 0.3)

    lo = max(0.0, np.floor(min(span, default=0.0)) - 0.5)
    hi = np.ceil(max(span, default=24.0)) + 0.5
    clipped = any(a > hi for a in acq_ends)
    axT.set_xlim(lo, hi)
    step, ticks = tick_ladder(lo, hi)
    axT.set_xticks(ticks)
    # 9h, 21h - the clock hour, not a duration. The day counter appears once a session
    # has run past midnight, where the hour alone is ambiguous; at a whole-day tick
    # spacing every label would read 0h, so only the day is shown.
    axT.set_xticklabels(
        [
            ("" if step % 24 == 0 else f"{int(t) % 24:d}h")
            + (f"\n+{int(t) // 24}d" if t >= 24 else "")
            for t in ticks
        ],
        fontsize=8,
    )
    for boundary in np.arange(24, hi, 24):  # where the clock rolls over
        axT.axvline(boundary, color="0.75", linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    axT.set_xlim(lo, hi)  # ticks and lines can each nudge the view; this one is final
    axT.set_xlabel("time of day, from the start of each session's own day", fontsize=9)
    axT.grid(axis="x", color="0.90", linewidth=0.6, zorder=0)
    axT.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        axT.spines[sp].set_visible(False)

    axB.set_xlabel("hours recorded", fontsize=9)
    axB.grid(axis="x", color="0.90", linewidth=0.6, zorder=0)
    axB.set_axisbelow(True)
    longest = max(
        (st["recorded_s"] for s in plotted for st in s["streams"]), default=0.0
    ) / 3600
    axB.set_xlim(0, max(0.5, longest * 1.55))
    for sp in ("top", "right", "left"):
        axB.spines[sp].set_visible(False)
    axB.tick_params(axis="x", labelsize=8)
    for ax in (axT, axB):  # the row labels are the y axis; the dashes add nothing
        ax.tick_params(axis="y", length=0)

    empty = sum(1 for s in plotted if s["recorded_s"] == 0)
    on = sum(st["recorded_s"] for s in plotted for st in s["streams"])
    off = sum(st["offline_s"] for s in plotted for st in s["streams"])
    cover = 100 * on / (on + off) if on + off > 0 else None
    pats = sorted({s["patient"] for s in plotted})
    total = sum(s["recorded_s"] for s in plotted)
    # One line only, on the left axis: a second title over the right panel collided
    # with it as soon as the folder name and the totals grew.
    axT.set_title(
        f"{folder_name}  -  {n} session{'s' if n != 1 else ''}, "
        f"{', '.join(pats) if len(pats) <= 3 else f'{len(pats)} patients'}, "
        f"{fmt_hms(total)} recorded"
        + (f", {cover:.0f}% coverage" if cover is not None else "")
        + (f", {empty} empty" if empty else ""),
        fontsize=11, loc="left", pad=24,
    )

    handles = [plt.Rectangle((0, 0), 1, 1, color=STREAM_COLOUR.get(k.split()[0], STREAM_COLOUR[""]))
               for k in kinds]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#d62728", alpha=0.30))
    handles.append(plt.Rectangle((0, 0), 1, 1, color="0.5", alpha=0.10, ec="0.6"))
    labels = kinds + [f"stop over {fmt_hm(pause_s)}", "acquisition open"]
    axT.legend(handles, labels, fontsize=8,
               ncol=min(len(labels), 6), frameon=False,
               loc="lower left", bbox_to_anchor=(0, 1.005))

    # Below the axes so bbox_inches="tight" grows the canvas instead of overprinting.
    # The offset is a fixed distance in INCHES turned into a figure fraction - as a bare
    # fraction it clears the x label on a tall figure and lands on top of it on a short
    # one, which is exactly the case a two-session folder produces.
    fig.text(
        0.005, -0.55 / fig.get_figheight(),
        f"bars are samples on disk; grey band is the file open but paused"
        f"{', clipped where it runs past the data' if clipped else ''}  |  "
        f"{src}  |  {Path(__file__).name}  {datetime.now():%Y-%m-%d %H:%M}",
        fontsize=6.5, color="0.55", ha="left", va="top",
    )
    save_png(fig, path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


# ------------------------------------------------------------------------------ main

SESSION_COLS = [
    "folder", "patient", "session", "start", "end", "recorded_s", "recorded_hms",
    "recorded_by_stream", "span_s", "acq_open_s", "n_segments", "n_channels",
    "n_ncs_files", "fs_hz", "families", "gb", "n_nev", "n_spike_files", "notes", "path",
]
STREAM_COLS = [
    "folder", "patient", "session", "stream", "fs", "start", "end",
    "recorded_s", "recorded_hms", "offline_hms", "coverage_pct",
    "n_segments", "n_pauses", "n_long_pauses", "longest_pause_hms",
    "n_channels", "families",
]
PAUSE_COLS = [
    "folder", "patient", "session", "stream", "start", "stop",
    "duration_s", "duration_hms", "prolonged", "where",
]
SEGMENT_COLS = [
    "folder", "patient", "session", "stream", "split", "file", "family", "fs",
    "n_records", "start", "end", "recorded_s", "recorded_hms", "span_s", "gap_s", "note",
]


def session_rows(sessions: list[dict]) -> list[dict]:
    out = []
    for s in sessions:
        r = {k: s.get(k) for k in SESSION_COLS if k != "recorded_hms"}
        r["recorded_hms"] = fmt_hms(s["recorded_s"])
        for k in ("start", "end"):
            r[k] = f"{s[k]:%Y-%m-%d %H:%M:%S}" if s[k] else ""
        for k in ("recorded_s", "span_s", "acq_open_s"):
            r[k] = round(s[k], 1)
        out.append(r)
    return out


def stream_rows(sessions: list[dict], pause_s: float = PAUSE_S) -> list[dict]:
    out = []
    for s in sessions:
        for st in s["streams"]:
            long_p = [p for p in st["pauses"] if p["seconds"] >= pause_s]
            out.append({
                "folder": s["folder"], "patient": s["patient"], "session": s["session"],
                "stream": st["label"], "fs": st["fs"],
                "start": f"{st['start']:%Y-%m-%d %H:%M:%S}" if st["start"] else "",
                "end": f"{st['end']:%Y-%m-%d %H:%M:%S}" if st["end"] else "",
                "recorded_s": round(st["recorded_s"], 1),
                "recorded_hms": fmt_hms(st["recorded_s"]),
                "offline_hms": fmt_hms(st["offline_s"]),
                "coverage_pct": st["coverage_pct"],
                "n_segments": st["n_segments"],
                "n_pauses": len(st["pauses"]), "n_long_pauses": len(long_p),
                "longest_pause_hms": fmt_hms(max((p["seconds"] for p in st["pauses"]), default=0)),
                "n_channels": st["n_channels"],
                "families": st["families"],
            })
    return out


def pause_rows(sessions: list[dict], pause_s: float = PAUSE_S) -> list[dict]:
    out = []
    for s in sessions:
        for st in s["streams"]:
            for p in st["pauses"]:
                out.append({
                    "folder": s["folder"], "patient": s["patient"],
                    "session": s["session"], "stream": st["label"],
                    "start": f"{p['start']:%Y-%m-%d %H:%M:%S}",
                    "stop": f"{p['end']:%Y-%m-%d %H:%M:%S}",
                    "duration_s": round(p["seconds"], 1),
                    "duration_hms": fmt_hms(p["seconds"]),
                    "prolonged": "yes" if p["seconds"] >= pause_s else "",
                    "where": p["where"],
                })
    return out


def segment_rows(sessions: list[dict]) -> list[dict]:
    out = []
    for s in sessions:
        for st in s["streams"]:
            for g in st["segments"]:
                out.append({
                    "folder": s["folder"], "patient": s["patient"], "session": s["session"],
                    "stream": st["label"],
                    "split": g["split"], "file": g["file"], "family": g["family"],
                    "fs": int(g["fs"]) if g["fs"] else "", "n_records": g["n_records"],
                    "start": f"{g['start']:%Y-%m-%d %H:%M:%S}" if g["start"] else "",
                    "end": f"{g['end']:%Y-%m-%d %H:%M:%S}" if g["end"] else "",
                    "recorded_s": round(g["recorded_s"], 1),
                    "recorded_hms": fmt_hms(g["recorded_s"]),
                    "span_s": round(g["span_s"], 1),
                    "gap_s": round(g["gap_s"], 1),
                    "note": g["note"],
                })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None,
                    help="MICROEPI directory; found next to this script if not given")
    ap.add_argument("--outdir", type=Path, default=OUT, help="where figures and CSVs go")
    ap.add_argument("--folder", action="append", default=None,
                    help="substring of the MicroEPI-B-## folders to keep; repeatable")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the session directories found, read nothing")
    ap.add_argument("--pause-threshold", type=float, default=PAUSE_S / 60, metavar="MIN",
                    help="a stop this many minutes long is called prolonged (default 15)")
    ap.add_argument("--gap-min", type=float, default=GAP_MIN_S, metavar="SEC",
                    help="shortest stop looked for at all, in seconds (default 60)")
    ap.add_argument("--use-cache", action="store_true",
                    help="replot from _scan_cache.json instead of re-reading the share")
    args = ap.parse_args(argv)
    args.pause_threshold *= 60
    args.root = resolve_data_root(args.root)
    print(f"MICROEPI: {args.root}")

    folders = sorted(p for p in args.root.glob(FOLDER_GLOB) if p.is_dir())
    if args.folder:
        folders = [f for f in folders if any(k.lower() in f.name.lower() for k in args.folder)]
    if not folders:
        print(f"no {FOLDER_GLOB} folders under {args.root}")
        return 1

    cache_path = args.outdir / "_scan_cache.json"

    if args.dry_run:
        for f in folders:
            print(f"\n{f.name}")
            for sdir, files in walk_sessions(f):
                n = sum(1 for e in files if e.name.lower().endswith(".ncs"))
                print(f"  {n:>5} .ncs  {sdir.relative_to(f).as_posix()}")
        return 0

    scanned: dict[str, list[dict]] = {}
    if args.use_cache:
        if not cache_path.exists():
            print(f"no cache at {cache_path} - run without --use-cache first")
            return 1
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        scanned = {k: [_dec(s) for s in v] for k, v in raw.items()}
        if any("streams" not in s for v in scanned.values() for s in v):
            print(f"{cache_path} was written before micro and macro were kept apart"
                  " - re-run without --use-cache")
            return 1
        print(f"replotting from {cache_path}")
    else:
        for f in folders:
            scanned[f.name] = scan_folder(f, args.gap_min, args.pause_threshold)
        save_bytes(
            json.dumps(scanned, default=_enc, indent=1).encode("utf-8"), cache_path
        )

    print()
    all_sessions: list[dict] = []
    for f in folders:
        sessions = scanned.get(f.name, [])
        all_sessions += sessions
        if not sessions:
            continue
        plot_folder(f.name, sessions, args.root / f.name,
                    args.outdir / f"{f.name}_durations.png", args.pause_threshold)
        save_csv(session_rows(sessions), SESSION_COLS, args.outdir / f"{f.name}_sessions.csv")
        save_csv(stream_rows(sessions, args.pause_threshold), STREAM_COLS,
                 args.outdir / f"{f.name}_streams.csv")
        save_csv(pause_rows(sessions, args.pause_threshold), PAUSE_COLS,
                 args.outdir / f"{f.name}_pauses.csv")
        save_csv(segment_rows(sessions), SEGMENT_COLS, args.outdir / f"{f.name}_segments.csv")

    if all_sessions:
        save_csv(session_rows(all_sessions), SESSION_COLS, args.outdir / "microepi_nlx_sessions.csv")
        save_csv(stream_rows(all_sessions, args.pause_threshold), STREAM_COLS,
                 args.outdir / "microepi_nlx_streams.csv")
        save_csv(pause_rows(all_sessions, args.pause_threshold), PAUSE_COLS,
                 args.outdir / "microepi_nlx_pauses.csv")
        save_csv(segment_rows(all_sessions), SEGMENT_COLS, args.outdir / "microepi_nlx_segments.csv")

    report(all_sessions, args.pause_threshold)
    print(f"\n  -> {args.outdir}")
    return 0


def report(sessions: list[dict], pause_s: float) -> None:
    """Per patient: online, offline, coverage, and the stops worth explaining.

    The same statistics the MicroEPI-G Blackrock report prints, per PATIENT rather than
    per folder, because EL043 sits in both B-01 and B-03 and a per-folder total would
    count it twice. Coverage is per stream, so it is never diluted by the other one.
    """
    by_patient = defaultdict(list)
    for s in sessions:
        by_patient[(s["patient"], s["folder"])].append(s)

    for (patient, folder), ss in sorted(by_patient.items()):
        on = sum(st["recorded_s"] for s in ss for st in s["streams"])
        off = sum(st["offline_s"] for s in ss for st in s["streams"])
        longs = [
            (s, st, p)
            for s in ss for st in s["streams"] for p in st["pauses"]
            if p["seconds"] >= pause_s
        ]
        print("_" * 96)
        print(f"\n  {patient} in {folder} - RECORDING STATISTICS\n")
        print(f"    Sessions                : {len(ss)}"
              f"   ({sum(1 for s in ss if s['recorded_s'] == 0)} with no data)")
        print(f"    Total time online       : {fmt_hms(on)}")
        print(f"    Total time offline      : {fmt_hms(off)}")
        print(f"    Percent coverage        : {100 * on / (on + off):.1f}%"
              if on + off > 0 else "    Percent coverage        : -")
        for st_label in sorted({st["label"] for s in ss for st in s["streams"]}):
            r = sum(st["recorded_s"] for s in ss for st in s["streams"] if st["label"] == st_label)
            print(f"      {st_label:<14}      {fmt_hms(r)}")
        print(f"\n    Prolonged stops (> {fmt_hm(pause_s)}): {len(longs)}")
        for s, st, p in sorted(longs, key=lambda x: x[2]["start"])[:40]:
            print(f"      {p['start']:%Y-%m-%d %H:%M}  {fmt_hms(p['seconds']):>9}"
                  f"  {st['label']:<13} {p['where']:<15} {short(s['label'], 40)}")
        if len(longs) > 40:
            print(f"      ... {len(longs) - 40} more, all of them in the pauses CSV")
    print("_" * 96)


if __name__ == "__main__":
    sys.exit(main())
