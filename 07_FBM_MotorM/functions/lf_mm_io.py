"""Raw readers for Motor Mapping: Blackrock NSx/NEV and Micromed TRC, no .mat, no library.

Both formats are documented and simple - a header, then interleaved int16 - so the whole
pipeline reads what the amplifiers wrote rather than an export someone has to regenerate.
Everything here was verified against real files on 2026-09-10.

THREE TRAPS THAT COST HALF A DAY EACH, all handled here so nothing downstream meets them.

  DECIMATE WITH OVERLAP. A channel is read in 32 MB chunks; decimating each chunk on its
  own zero-pads beyond its edges, and a signal with a DC level rings there - the
  photodiode showed a spike to 808 and 1805 around a 1600 cue level every 10.965 s
  (32 MB / (51 ch x 2 B) = 328,965 samples), and every micro channel a smaller one. They
  are not in the raw file. pull_channels now decimates each chunk with the tail of the
  previous and the head of the next and keeps only its own samples (found 2026-09-12).

  TRC LABELS NEED THE ORDER ZONE. LABCOD holds 640 fixed slots whatever the channel count,
  and the ORDER zone maps acquisition index -> slot. Reading LABCOD slot i as channel i
  gives labels that are correct for the first handful of channels and silently wrong after
  that; it is how MKR1+ came to be read as an EEG contact. read_trc_meta always goes
  through ORDER.

  NSX PARTS EACH RESTART AT ZERO. A split recording's parts all carry the same session
  time origin in the basic header and a data-packet timestamp of 0, so the session timeline
  is the CONCATENATION of the parts, not a set of offsets. session_timeline() builds that,
  and every time this package quotes is "seconds since the first part began".

  ...AND THE PARTS ARE NOT CONTIGUOUS (found 2026-09-11). Central closes one part and
  opens the next with 4-6 s of nothing in between: for G-05 the header origins say 6.5 s
  and 4.8 s, the sync-pulse cadence says 6.34 s and 4.81 s, and the photodiode lost one
  whole trial in each gap. The concatenated clock is still right for anything inside one
  part - cue and LFP are cut from the same concatenation - but an epoch that spans a seam
  has a jump in it (crosses_splice), and a Micromed offset measured on one part is wrong
  by the gap on the next (11_mm_sync.py measures one per part). The header origin is
  kept on every part for exactly this: it is the only absolute time Blackrock writes.
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
    # overlap for the decimation filter: resample_poly's default FIR reaches 10*step
    # samples either side, so 32*step of context on each edge leaves no transient inside
    # the kept samples. A multiple of step, so the output grid is the same as without it.
    pad = 32 * step if step > 1 else 0

    def emit(prev_tail, cur, next_head):
        """Decimate cur with its neighbours' edges for context; keep cur's own samples."""
        for i in idxs:
            col = cur[:, i].astype(np.float32)
            if step == 1:
                acc[i].append(col)
                continue
            head = prev_tail[:, i].astype(np.float32) if prev_tail is not None else None
            tail = next_head[:, i].astype(np.float32) if next_head is not None else None
            ext = np.concatenate([v for v in (head, col, tail) if v is not None])
            # 'line' pads the file's own first and last edge with a linear extension,
            # so a channel sitting on a DC level does not ring there either
            y = resample_poly(ext, 1, step, padtype="line")
            k0 = 0 if head is None else len(head) // step
            acc[i].append(y[k0:k0 + int(np.ceil(len(col) / step))])

    done = 0
    prev_tail, cur = None, None
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
            if cur is not None:                       # one chunk of lookahead
                emit(prev_tail, cur, a[:pad] if pad else None)
                prev_tail = cur[-pad:] if pad else None
            cur = a
            left -= k
            done += k
            if progress:
                progress(done, meta["n_samp"])
    if cur is not None:
        emit(prev_tail, cur, None)
    return {i: np.concatenate(v) for i, v in acc.items()}, fs / step


def part_paths(pre):
    """The .ns6 parts a preset names, first..last inclusive, in order."""
    folder = pre["blackrock_dir"]
    first, last = pre["blackrock"]
    stem, a = first.rsplit("-", 1)
    b = last.rsplit("-", 1)[1]
    return [os.path.join(folder, f"{stem}-{n}.ns6") for n in range(int(a), int(b) + 1)]


def real_parts(parts, min_s=1.0):
    """The parts that hold data. Central writes a 10 ms stub after every real part."""
    return [p for p in parts if p["dur"] >= min_s]


def part_num(part):
    """'20250120-181120-713.ns6' -> '713', '20250618-145815-1393.ns6' -> '1393'. By the
    last hyphen, not a fixed slice: G-01 and G-02 have 3-digit parts, G-03 and G-05 four."""
    return os.path.splitext(os.path.basename(part["name"]))[0].rsplit("-", 1)[1]


def _origin_s(part):
    import datetime as dt
    return dt.datetime.strptime(part["origin"], "%Y-%m-%d %H:%M:%S.%f").timestamp()


def gaps(parts, min_gap_s=0.5):
    """[(part a, part b, gap_s)] for consecutive real parts whose header origins say the
    recording stopped between them. Not every boundary is a gap: a fixed-length split
    (G-02, G-03: 301.155 s parts, origins contiguous to 0.07 s) is one continuous
    recording and an epoch cut across it is sound. A stop/start (G-05: 10 ms stubs,
    4-6.5 s gaps) is a seam. Origins carry ~0.2 s of error; every real gap seen is >= 4 s."""
    rp = real_parts(parts)
    out = []
    for a, b in zip(rp[:-1], rp[1:]):
        g = _origin_s(b) - (_origin_s(a) + a["dur"])
        if abs(g) > min_gap_s:
            out.append((a, b, g))
    return out


def splices(parts):
    """Session-clock times of the seams: where the next real part begins after a gap."""
    return [b["t0"] for _a, b, _g in gaps(parts)]


def crosses_splice(t, window, parts):
    """True for every t whose epoch [t+window[0], t+window[1]] contains a seam.

    The seam is the whole stretch from the end of one real part's data to the next real
    part's t0: the 10 ms stub between them was written from inside the gap, so an epoch
    that only reaches into the stub already holds the jump."""
    t = np.asarray(t, float)
    hit = np.zeros(t.shape, bool)
    for a, b, _g in gaps(parts):
        hit |= (t + window[0] < b["t0"]) & (t + window[1] > a["t0"] + a["dur"])
    return hit


def part_index(t, parts):
    """Which real part each session-clock time falls in (index into real_parts)."""
    rp = real_parts(parts)
    t0 = np.array([p["t0"] for p in rp])
    return np.clip(np.searchsorted(t0, np.asarray(t, float), side="right") - 1, 0, len(rp) - 1)


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
    """Written under a temporary name and renamed into place: the share truncates a file
    whose write fails, and a cache is large enough for that to happen. The old cache is
    replaced only once the new one is complete."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    if os.path.getsize(tmp) < arr.nbytes:
        os.remove(tmp)
        raise IOError(f"short write for {path}: the share truncated the cache")
    os.replace(tmp, path)
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_cache(path):
    j = os.path.splitext(path)[0] + ".json"
    if not (os.path.exists(path) and os.path.exists(j)):
        return None, None
    with open(j, encoding="utf-8") as f:
        return np.load(path), json.load(f)
