"""Raw readers for Motor Mapping: Blackrock NSx/NEV and Micromed TRC, no .mat, no library.

Both formats are documented and simple - a header, then interleaved int16 - so the whole
pipeline reads what the amplifiers wrote rather than an export someone has to regenerate.
Everything here was verified against real files on 2026-09-10.

TWO TRAPS THAT COST HALF A DAY, both now handled here so nothing downstream meets them.

  TRC LABELS NEED THE ORDER ZONE. LABCOD holds 640 fixed slots whatever the channel count,
  and the ORDER zone maps acquisition index -> slot. Reading LABCOD slot i as channel i
  gives labels that are correct for the first handful of channels and silently wrong after
  that; it is how MKR1+ came to be read as an EEG contact. read_trc_meta always goes
  through ORDER.

  NSX PARTS EACH RESTART AT ZERO. A split recording's parts all carry the same session
  time origin in the basic header and a data-packet timestamp of 0, so the session timeline
  is the CONCATENATION of the parts, not a set of offsets. session_timeline() builds that,
  and every time this package quotes is "seconds since the first part began".
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np

# ---------------------------------------------------------------------------
# Blackrock NSx (.ns6)
# ---------------------------------------------------------------------------
def read_nsx_meta(path: str) -> dict:
    """Headers only - a few kB - so choosing a channel never costs a 900 MB read."""
    with open(path, "rb") as f:
        head = f.read(314)
        n_hdr = struct.unpack("<I", head[10:14])[0]
        period, tres = struct.unpack("<II", head[286:294])
        origin = struct.unpack("<8H", head[294:310])
        n_chan = struct.unpack("<I", head[310:314])[0]
        ext = f.read(66 * n_chan)
        f.seek(n_hdr)
        f.read(1)                                   # packet tag 0x01
        ts, npts = struct.unpack("<II", f.read(8))
    chans = []
    for i in range(n_chan):
        r = ext[i * 66:(i + 1) * 66]
        if len(r) < 66:
            break
        chans.append(dict(
            idx=i, id=struct.unpack("<H", r[2:4])[0],
            label=r[4:20].rstrip(b"\x00 ").decode("latin-1", "replace"),
            units=r[30:46].rstrip(b"\x00 ").decode("latin-1", "replace")))
    fs = float(tres) / float(period) if period else float("nan")
    size = os.path.getsize(path)
    n_avail = min(npts, (size - n_hdr - 9) // (n_chan * 2))
    y, mo, _dw, d, hh, mi, ss, ms = origin
    return dict(path=path, fs=fs, n_chan=n_chan, chans=chans,
                names=[c["label"] for c in chans],
                data_off=n_hdr + 9, n_bytes=2, n_samp=int(n_avail),
                dur=n_avail / fs if fs else 0.0,
                origin=f"{y:04d}-{mo:02d}-{d:02d} {hh:02d}:{mi:02d}:{ss:02d}.{ms:03d}")


def read_nev_digital(path: str) -> dict:
    """The .nev digital/serial stream.

    NOTE these are SYNCHRONISATION words, not trials. They arrive on a regular ~5.78 s
    cadence carrying a 4-byte code, which looks exactly like a trial marker and is not one.
    Trial times come from the photodiode. Returned so the cadence can be SHOWN rather than
    quietly mistaken for the task.
    """
    with open(path, "rb") as f:
        hd = f.read(336)
        n_hdr, n_pkt = struct.unpack("<II", hd[12:20])
        tres = struct.unpack("<I", hd[20:24])[0]
        size = os.path.getsize(path)
        n_packets = (size - n_hdr) // n_pkt
        f.seek(n_hdr)
        raw = f.read(n_packets * n_pkt)
    a = np.frombuffer(raw[:(len(raw) // n_pkt) * n_pkt], dtype=np.uint8).reshape(-1, n_pkt)
    ts = a[:, 0:4].copy().view(np.uint32).ravel()
    pid = a[:, 4:6].copy().view(np.uint16).ravel()
    dig = pid == 0
    return dict(fs=float(tres), n_packets=int(n_packets),
                n_digital=int(dig.sum()), n_spike=int((~dig).sum()),
                t=ts[dig] / float(tres), value=a[dig, 8:10].copy().view(np.uint16).ravel(),
                spike_ids=np.unique(pid[~dig]) if (~dig).any() else np.array([]))


# ---------------------------------------------------------------------------
# Micromed TRC
# ---------------------------------------------------------------------------
def read_trc_meta(path: str) -> dict:
    with open(path, "rb") as f:
        f.seek(138)
        ds, nch, _mux, fs, nb = struct.unpack("<IHHHH", f.read(12))
        f.seek(176)
        blob = f.read(16 * 20)
        z = {}
        for i in range(20):
            r = blob[i * 16:(i + 1) * 16]
            n = r[0:8].rstrip(b"\x00 ").decode("latin-1", "replace")
            off, ln = struct.unpack("<II", r[8:16])
            if n:
                z[n] = (off, ln)
        f.seek(*z["ORDER"][:1])
        order = np.frombuffer(f.read(z["ORDER"][1]), dtype="<u2")
        f.seek(z["LABCOD"][0])
        lab = f.read(z["LABCOD"][1])
        notes = []
        if "NOTE" in z:
            f.seek(z["NOTE"][0])
            nraw = f.read(z["NOTE"][1])
            for i in range(z["NOTE"][1] // 44):
                rec = nraw[i * 44:(i + 1) * 44]
                s = struct.unpack("<I", rec[0:4])[0]
                t = rec[4:44].rstrip(b"\x00 ").decode("latin-1", "replace").strip()
                if t and s:
                    notes.append((s / float(fs), t))
    names = [lab[int(order[i]) * 128:int(order[i]) * 128 + 128][2:8]
             .rstrip(b"\x00 ").decode("latin-1", "replace") for i in range(nch)]
    n_samp = (os.path.getsize(path) - ds) // (nch * nb)
    return dict(path=path, fs=float(fs), n_chan=nch, names=names, notes=notes,
                data_off=ds, n_bytes=nb, n_samp=int(n_samp), dur=n_samp / float(fs),
                dtype="<u2")


# ---------------------------------------------------------------------------
# shared: pull channels out of an interleaved file
# ---------------------------------------------------------------------------
def pull_channels(meta: dict, idxs, target_fs=None, dtype=None, chunk_mb=32,
                  progress=None):
    """Several channels in ONE pass, optionally decimated. Returns (dict, fs_out).

    One pass matters: a channel is one column of an interleaved file, so reading it costs
    the whole file whether you want one channel or twenty.
    """
    from scipy.signal import resample_poly
    nch, nb, fs = meta["n_chan"], meta["n_bytes"], meta["fs"]
    dtype = dtype or meta.get("dtype", "<i2")
    step = max(1, int(round(fs / target_fs))) if target_fs else 1
    rows = max(nch, (chunk_mb * 1024 * 1024) // (nch * nb))
    rows -= rows % step
    idxs = list(idxs)
    acc = {i: [] for i in idxs}
    done = 0
    with open(meta["path"], "rb") as f:
        f.seek(meta["data_off"])
        left = meta["n_samp"]
        while left > 0:
            k = min(rows, left)
            buf = f.read(k * nch * nb)
            if not buf:
                break
            buf = buf[:len(buf) - (len(buf) % (nch * nb))]
            a = np.frombuffer(buf, dtype=dtype).reshape(-1, nch)
            for i in idxs:
                col = a[:, i].astype(np.float32)
                acc[i].append(resample_poly(col, 1, step) if step > 1 else col)
            left -= k
            done += k
            if progress:
                progress(done, meta["n_samp"])
    return {i: np.concatenate(v) for i, v in acc.items()}, fs / step


def session_timeline(part_paths):
    """Per-part durations and their cumulative starts on the session clock.

    Every part restarts its own timestamp at zero, so "seconds since the first part began"
    is the only unambiguous way to quote a Blackrock time across a split recording.
    """
    out, t0 = [], 0.0
    for p in part_paths:
        if not os.path.exists(p):
            continue
        m = read_nsx_meta(p)
        out.append(dict(path=p, name=os.path.basename(p), t0=t0, dur=m["dur"],
                        fs=m["fs"], n_chan=m["n_chan"], origin=m["origin"]))
        t0 += m["dur"]
    return out, t0


def find_channel(names, wanted) -> int | None:
    low = [n.lower() for n in names]
    for w in ([wanted] if isinstance(wanted, str) else wanted):
        if w.lower() in low:
            return low.index(w.lower())
    return None


def save_cache(path, arr, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_cache(path):
    j = os.path.splitext(path)[0] + ".json"
    if not (os.path.exists(path) and os.path.exists(j)):
        return None, None
    with open(j, encoding="utf-8") as f:
        return np.load(path), json.load(f)
