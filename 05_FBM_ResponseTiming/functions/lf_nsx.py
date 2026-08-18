"""
lf_nsx.py - minimal reader for Blackrock NSx files (.ns2 / .ns5 / .ns6).

Why this exists: the MicroEPI micro-electrode side is recorded on Blackrock, and
Blackrock's ANALOG INPUTS are named `ainp1..ainp16`. If a microphone was patched
into one of them, the speech signal is sitting in `raw_blackrock/*.ns6` at 30 kHz
- which is the only place in this whole dataset that could carry audio, and the
only route to a real speech-onset time. Everything else is locked to the GO cue.

The curated `*_export_Labs_*.mat` files do NOT contain the ainp channels (checked
on all six patients), so they cannot answer the question; the raw NSx must be read
directly. No Blackrock reader (brpylib / neo) is installed, and the format is
simple enough that a dependency is not worth it.

Format (NSx 2.2 / 2.3, per Blackrock's file spec):
    bytes  0..7    'NEURALCD'
           8..9    file spec major, minor
          10..13   bytes in headers  (= offset of the first data packet)
          14..29   label, e.g. '30 kS/s'
         286..289  period            (sampling = clock / period)
         290..293  clock  (time resolution of timestamps, usually 30000)
         310..313  channel count
    then one extended header per channel; its size is derived rather than
    assumed, as (bytes_in_headers - 314) / channel_count, which self-validates.
    Each extended header holds a 16-byte label at offset 4.

    Data packets: 0x01, uint32 timestamp, uint32 n_samples, then int16 samples
    interleaved channel-major (all channels for sample 0, then sample 1, ...).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HDR_FIXED = 314


@dataclass
class NsxHeader:
    path: Path
    file_spec: tuple
    bytes_in_headers: int
    label: str
    period: int
    clock: int
    fs: float
    n_channels: int
    chan_labels: list
    chan_ids: list
    ext_size: int
    data_offset: int
    n_samples: int
    timestamp: int


def read_header(path) -> NsxHeader:
    p = Path(path)
    with open(p, "rb") as fh:
        magic = fh.read(8)
        if magic not in (b"NEURALCD", b"NEURALSG"):
            raise ValueError(f"{p.name}: not an NSx file (magic {magic!r})")
        if magic == b"NEURALSG":
            raise NotImplementedError(f"{p.name}: NSx 2.1 has no channel labels")
        spec = struct.unpack("<BB", fh.read(2))
        bih = struct.unpack("<I", fh.read(4))[0]
        label = fh.read(16).split(b"\x00")[0].decode("latin-1")
        fh.read(256)                                   # comment
        period = struct.unpack("<I", fh.read(4))[0]
        clock = struct.unpack("<I", fh.read(4))[0]
        fh.read(16)                                    # time origin
        n_ch = struct.unpack("<I", fh.read(4))[0]

        ext_size = (bih - HDR_FIXED) // max(n_ch, 1)
        if ext_size <= 0:
            raise ValueError(f"{p.name}: bad extended-header size {ext_size}")
        labels, ids = [], []
        for _ in range(n_ch):
            blk = fh.read(ext_size)
            ids.append(struct.unpack_from("<H", blk, 2)[0])
            labels.append(blk[4:20].split(b"\x00")[0].decode("latin-1").strip())

        fh.seek(bih)
        tag = fh.read(1)
        ts, nsamp = 0, 0
        if tag == b"\x01":
            if spec[0] >= 3:
                ts = struct.unpack("<Q", fh.read(8))[0]
            else:
                ts = struct.unpack("<I", fh.read(4))[0]
            nsamp = struct.unpack("<I", fh.read(4))[0]
        data_offset = fh.tell()

    fs = float(clock) / float(period or 1)
    return NsxHeader(p, spec, bih, label, period, clock, fs, n_ch, labels, ids,
                     ext_size, data_offset, nsamp, ts)


def find_channels(h: NsxHeader, pattern: str = "ainp") -> list:
    """Indices of channels whose label matches, case-insensitively."""
    pat = pattern.lower()
    return [i for i, l in enumerate(h.chan_labels) if pat in l.lower()]


def read_window(h: NsxHeader, chan_idx, t0_s: float, t1_s: float,
                max_samples: int = 6_000_000) -> np.ndarray:
    """Read [t0_s, t1_s) for the given channel indices -> (n_chan, n_samp) int16.

    Seeks rather than reading the whole file: an .ns6 here is ~920 MB and a task
    block is a few minutes of it.
    """
    chan_idx = list(chan_idx)
    i0 = max(0, int(round(t0_s * h.fs)))
    i1 = min(h.n_samples, int(round(t1_s * h.fs)))
    if i1 <= i0:
        return np.empty((len(chan_idx), 0), dtype=np.int16)
    n = i1 - i0
    step = 1
    if n > max_samples:                      # decimate by striding whole frames
        step = int(np.ceil(n / max_samples))
    frame = h.n_channels * 2                 # bytes per sample across all channels
    out = np.empty((len(chan_idx), len(range(0, n, step))), dtype=np.int16)
    with open(h.path, "rb") as fh:
        for k, s in enumerate(range(0, n, step)):
            fh.seek(h.data_offset + (i0 + s) * frame)
            buf = np.frombuffer(fh.read(frame), dtype="<i2")
            if buf.size < h.n_channels:
                out = out[:, :k]
                break
            out[:, k] = buf[chan_idx]
    return out


def effective_fs(h: NsxHeader, t0_s: float, t1_s: float,
                 max_samples: int = 6_000_000) -> float:
    """Sampling rate actually returned by read_window after any striding."""
    n = max(0, int(round((t1_s - t0_s) * h.fs)))
    step = int(np.ceil(n / max_samples)) if n > max_samples else 1
    return h.fs / step


def describe(path) -> dict:
    h = read_header(path)
    ai = find_channels(h, "ainp")
    return {
        "file": Path(path).name,
        "fs": h.fs,
        "n_channels": h.n_channels,
        "n_samples": h.n_samples,
        "duration_s": h.n_samples / h.fs if h.fs else float("nan"),
        "ainp": [h.chan_labels[i] for i in ai],
        "ainp_idx": ai,
        "all_labels": h.chan_labels,
    }
