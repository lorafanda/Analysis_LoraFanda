"""
lf_micro_io.py
==============

Readable Blackrock NS6 session loader for the MicroEPI micro-electrode pipeline.

This replaces the opaque compiled ``lf_blackrock_io`` module with plain source.
It uses the same high-level reader that ``lf_io_utils`` relies on for NS6
(``neo``), but via ``neo.io.BlackrockIO`` so that signals come back **rescaled to
microvolts** — important because the spike pipeline's amplitude knobs
(``MAX_AMPLITUDE_UV``, ``AMP_MIN_UV`` ...) are in µV. (Note: ``lf_io_utils.load_ns6``
uses ``BlackrockRawIO``/``get_analogsignal_chunk`` which returns *raw int16 ADC
counts*, not µV — building the session loader on that would silently break the µV
thresholds, which is why this module reads via BlackrockIO instead.)

Public API (matches the notebook's call sites)
----------------------------------------------
read_ns6_fragment(ns6_path, *, verbose=True) -> dict
read_ns6_files(paths, *, micro_channel_pattern=None,
               drop_micro_pattern=r'^[Xx](_?\\d+)?$', verbose=True) -> dict
read_blackrock_session(data_dir, base_filename, file_suffixes, *,
                       micro_channel_pattern=None,
                       drop_micro_pattern=r'^[Xx](_?\\d+)?$', verbose=True) -> dict
extract_analog_channel(d_session, channel_name) -> ndarray
"""
from __future__ import annotations

import os
import re
import glob
import numpy as np

__all__ = ["read_ns6_fragment", "read_ns6_files",
           "read_blackrock_session", "extract_analog_channel"]


def read_ns6_fragment(ns6_path, *, verbose=True):
    """Read one Blackrock NS6 file -> dict(signals[µV float32], fs, names, n_samples, duration_s, path).

    Signals are returned in **microvolts**, shape (n_samples, n_channels).
    """
    import neo.io  # lazy: importing this module never requires neo to be installed

    if not os.path.isfile(ns6_path):
        raise FileNotFoundError(ns6_path)

    reader = neo.io.BlackrockIO(filename=ns6_path)
    blk = reader.read_block(lazy=False)
    seg = blk.segments[0]
    if not seg.analogsignals:
        raise RuntimeError(f"No analog signals in {ns6_path}")

    fs = None
    parts, names = [], []
    for ana in seg.analogsignals:
        # sampling rate (Hz)
        try:
            f = float(np.asarray(ana.sampling_rate.rescale("Hz").magnitude).ravel()[0])
        except Exception:
            f = float(ana.sampling_rate)
        if fs is None:
            fs = f
        elif abs(f - fs) > 1e-6:
            raise ValueError(f"Mixed sample rates in {ns6_path}: {f} vs {fs}")

        # signal in microvolts
        try:
            arr = np.asarray(ana.rescale("uV").magnitude, dtype=np.float32)
        except Exception:
            arr = np.asarray(ana.magnitude, dtype=np.float32)  # already µV for BlackrockIO
        if arr.ndim == 1:
            arr = arr[:, None]
        parts.append(arr)

        # channel names
        nm = []
        if hasattr(ana, "array_annotations"):
            nm = list(ana.array_annotations.get("channel_names", []))
        if len(nm) != arr.shape[1]:
            nm = [f"ch{len(names) + k}" for k in range(arr.shape[1])]
        names.extend(str(x) for x in nm)

    signals = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
    n_samples = int(signals.shape[0])
    duration_s = n_samples / fs
    if verbose:
        print(f"  [ns6] {os.path.basename(ns6_path)}: {n_samples} samples @ {fs:g} Hz, "
              f"{duration_s:.1f} s, {signals.shape[1]} channels")
    return dict(signals=signals, fs=fs, names=names,
                n_samples=n_samples, duration_s=duration_s, path=ns6_path)


def read_ns6_files(paths, *, micro_channel_pattern=None,
                   drop_micro_pattern=r"^[Xx](_?\d+)?$", verbose=True):
    """Read + concatenate an explicit, ordered list of NS6 file paths into one session.

    Same return contract as :func:`read_blackrock_session`. Use this when you just
    want to point at "a few NS6 files" directly (any patient), without the
    base-filename + suffix scheme.

    Channels whose names start with 'ainp' (case-insensitive) -> analog group
    (photodiode / sync); everything else -> micro group, unless
    ``micro_channel_pattern`` is given. Micros matching ``drop_micro_pattern``
    (e.g. unused 'X' wires) are dropped.
    """
    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError("read_ns6_files: empty path list")
    if verbose:
        print(f"[blackrock] reading {len(paths)} NS6 file(s)")

    # read fragments + consistency checks
    frags = [read_ns6_fragment(p, verbose=verbose) for p in paths]
    fs = frags[0]["fs"]
    names_ref = frags[0]["names"]
    for fi, fr in enumerate(frags):
        if abs(fr["fs"] - fs) > 1e-6:
            raise ValueError(f"Fragment {fi} fs={fr['fs']} differs from fs={fs}")
        if fr["names"] != names_ref:
            raise ValueError(f"Fragment {fi} has different channel names than fragment 0")

    # concatenate along time
    signals_all = np.concatenate([fr["signals"] for fr in frags], axis=0)
    n_samples = int(signals_all.shape[0])
    duration_s = n_samples / fs

    # fragment boundaries (seconds): [0, len0, len0+len1, ..., total] / fs
    cum = np.cumsum([0] + [fr["n_samples"] for fr in frags]).astype(np.int64)
    fragment_boundaries_s = (cum / fs).astype(np.float64)
    if verbose:
        print(f"[blackrock] concatenated: {n_samples} samples = {duration_s / 60:.1f} min")

    # split micro vs ainp
    is_ainp = np.array([str(n).lower().startswith("ainp") for n in names_ref], dtype=bool)
    if micro_channel_pattern:
        micro_re = re.compile(micro_channel_pattern)
        is_micro = np.array([bool(micro_re.search(str(n))) for n in names_ref], dtype=bool)
    else:
        is_micro = ~is_ainp

    # drop unused micros (e.g. 'X' wires)
    keep_mask = is_micro.copy()
    dropped = []
    if drop_micro_pattern:
        drop_re = re.compile(drop_micro_pattern, re.IGNORECASE)
        for ci, n in enumerate(names_ref):
            if is_micro[ci] and drop_re.match(str(n)):
                keep_mask[ci] = False
                dropped.append(str(n))

    micro_idx = np.flatnonzero(keep_mask)
    ainp_idx = np.flatnonzero(is_ainp)
    signals_micro = signals_all[:, micro_idx].astype(np.float32)
    signals_ainp = signals_all[:, ainp_idx].astype(np.float32)
    names_micro = [str(names_ref[i]) for i in micro_idx]
    names_ainp = [str(names_ref[i]) for i in ainp_idx]

    if verbose:
        if dropped:
            print(f"[blackrock] dropped unused micros ({len(dropped)}): {dropped}")
        print(f"[blackrock] micro channels ({len(names_micro)}): {names_micro}")
        print(f"[blackrock] ainp channels  ({len(names_ainp)}): {names_ainp}")

    return dict(
        signals_micro=signals_micro, signals_ainp=signals_ainp,
        names_micro=names_micro, names_ainp=names_ainp,
        fs=fs, n_samples=n_samples, duration_s=duration_s,
        fragment_boundaries_s=fragment_boundaries_s, fragment_paths=paths,
    )


def read_blackrock_session(data_dir, base_filename, file_suffixes, *,
                           micro_channel_pattern=None,
                           drop_micro_pattern=r"^[Xx](_?\d+)?$", verbose=True):
    """Load + concatenate NS6 fragments named ``{base_filename}{suffix}.ns6`` in
    ``data_dir`` (in the given suffix order) into one continuous session.

    Thin wrapper that resolves the fragment paths then defers to
    :func:`read_ns6_files` (same return contract).
    """
    paths = []
    for sfx in file_suffixes:
        candidate = os.path.join(data_dir, f"{base_filename}{sfx}.ns6")
        if os.path.isfile(candidate):
            paths.append(candidate)
            continue
        matches = sorted(glob.glob(os.path.join(data_dir, f"*{sfx}.ns6")))
        if matches:
            paths.append(matches[0])
        else:
            raise FileNotFoundError(f"No NS6 found for suffix '{sfx}': tried {candidate}")
    return read_ns6_files(paths, micro_channel_pattern=micro_channel_pattern,
                          drop_micro_pattern=drop_micro_pattern, verbose=verbose)


def extract_analog_channel(d_session, channel_name):
    """Return the named channel (µV float32, 1-D). Searches ainp then micro groups."""
    names_ainp = list(d_session.get("names_ainp", []))
    if channel_name in names_ainp:
        return np.asarray(d_session["signals_ainp"][:, names_ainp.index(channel_name)],
                          dtype=np.float32)
    names_micro = list(d_session.get("names_micro", []))
    if channel_name in names_micro:
        return np.asarray(d_session["signals_micro"][:, names_micro.index(channel_name)],
                          dtype=np.float32)
    raise ValueError(f"Channel '{channel_name}' not found "
                     f"(ainp={names_ainp}, micro n={len(names_micro)})")
