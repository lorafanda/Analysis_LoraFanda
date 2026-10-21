#!/usr/bin/env python3
"""microepi_spikes.py - tetrode spike sorting on the MicroEPI microwires, at 30 kHz.

    python microepi_spikes.py --patient G-03 --prepare
    python microepi_spikes.py --patient G-03 --sorter mountainsort5
    python microepi_spikes.py --patient G-03 --sorter spykingcircus2
    python microepi_spikes.py --patient G-03 --rasters

NEEDS ITS OWN ENVIRONMENT. Do not install this into LORA_ENV2:

    conda create -n spikesort python=3.11 -y
    conda activate spikesort
    pip install spikeinterface mountainsort5 hdbscan numba

`spikeinterface[full]` pulls pynwb, zarr, pynapple, cuda-python and more, and on the way
it upgrades numpy, scipy, pandas, scikit-learn and matplotlib. Run into the analysis env
on 2026-10-08 that took numpy from 1.26 to 2.4 while the compiled extensions around it
were still built for 1.x, and every import died on "numpy.dtype size changed". This
script needs numpy, scipy, neo and matplotlib only - no mne, no 140 - so it runs happily
in an environment of its own while the ERSP driver stays in LORA_ENV2.

WHAT IT DOES. The microwires come off the shaft in fours - 1-4, 5-8, 9-12 - so each group
is a tetrode and is sorted on its own. Wire 1 of the shaft is R1, the hardware reference,
and is subtracted from its shaft and dropped (--reref), which leaves the FIRST tetrode of
each shaft with three wires rather than four; the sorters handle that, but it is worth
knowing that tetrode 0 is a triode. Wires 5 and 9 are R2, also reference-wired and
noisier at times; they are kept as data by default and can be dropped with --drop-r2.

Two CPU sorters are offered so the units can be compared rather than trusted: MountainSort5
and SpyKING CIRCUS 2, both through SpikeInterface, both without a GPU.

TRIALS. Spike times are referenced to the same cue times the ERSP tree uses - 140's
curated trials, placed on the .ns6 clock by their own photodiode pulse
(lf_microepi_ns6.trials_on_ns6). Only G-03 and G-04 resolve cleanly today; the other four
do not match their trial tables and are refused rather than sorted against wrong times.

DISK. A shaft is staged to int16 at 30 kHz under outputs/microepi_spikes/<PID>/raw/ and
memory-mapped from there, written one .ns6 file at a time so peak memory stays near one
file rather than the whole session. Nothing in DATARAW is written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "functions"))

import lf_microepi_ns6 as NS                      # noqa: E402
from functions import config as cfg               # noqa: E402

TREE = "microepi_spikes"
FS = 30000.0
BAND = (300.0, 6000.0)        # spike band; 6 kHz is well inside the 15 kHz Nyquist
SORTERS = ("mountainsort5", "spykingcircus2")
# patients whose 140 trial table lines up with their .ns6 photodiode (see the dry run)
TRIALS_OK = ("G-03", "G-04")


def out_root(patient):
    return Path(cfg.outputs_root) / TREE / NS.SESSIONS[patient]["pat"]


def need_si():
    try:
        import spikeinterface.full as si          # noqa: F401
        import probeinterface as pi               # noqa: F401
        return si, pi
    except ImportError as e:
        raise SystemExit(
            f"{e}\n\nspike sorting needs:\n"
            '    pip install "spikeinterface[full]" mountainsort5 hdbscan numba\n')


def stage_shaft(patient, shaft, wires, reref_scheme, files=None, force=False):
    """Write one shaft's re-referenced traces to int16 at 30 kHz, a file at a time.

    Staged rather than held in memory: twelve channels of 30 kHz over half an hour is
    2.6 GB as float32 before anything is filtered, and the sorters want random access to
    the whole recording anyway. Returns (path, channel names, gain to uV).
    """
    d = out_root(patient) / "raw"
    d.mkdir(parents=True, exist_ok=True)
    dat = d / f"{shaft}.dat"
    meta = d / f"{shaft}.json"
    if dat.exists() and meta.exists() and not force:
        m = json.loads(meta.read_text())
        print(f"  {shaft}: staged already ({m['n_samples']} samples, "
              f"{len(m['names'])} ch) - use --force to redo")
        return dat, m["names"], m["gain"], m["n_samples"]

    S = NS.SESSIONS[patient]
    use = files if files is not None else S["files"]
    # one pass to fix the int16 gain, so every file is scaled the same way
    first = NS.read_ns6(patient, files=use[:1], want=wires, verbose=False)
    x0, names, _ = NS.reref(first["x"], first["names"], scheme=reref_scheme)
    peak = float(np.percentile(np.abs(x0), 99.99)) * 1.5 or 1.0
    gain = peak / 32000.0
    del first, x0

    n_total = 0
    with open(dat, "wb") as fh:
        for i in use:
            blk = NS.read_ns6(patient, files=[i], want=wires, verbose=False)
            x, names, _drop = NS.reref(blk["x"], blk["names"], scheme=reref_scheme)
            q = np.clip(np.round(x / gain), -32768, 32767).astype("int16")
            fh.write(q.tobytes(order="C"))        # interleaved, samples x channels
            n_total += q.shape[0]
            print(f"  {shaft}: staged {os.path.basename(str(i))} -> {n_total} samples",
                  flush=True)
            del blk, x, q
    meta.write_text(json.dumps(dict(names=names, gain=gain, fs=FS,
                                    n_samples=int(n_total), shaft=shaft)))
    return dat, names, gain, n_total


def recording_for(si, pi, dat, names, gain, n_samples, drop_r2=False):
    """A SpikeInterface recording for one shaft, with its tetrodes as groups."""
    rec = si.read_binary(str(dat), sampling_frequency=FS, num_channels=len(names),
                         dtype="int16", gain_to_uV=gain, offset_to_uV=0.0)
    rec = rec.rename_channels(names) if hasattr(rec, "rename_channels") else rec
    groups, _odd, ref_of = NS.tetrodes(names)
    if drop_r2:
        keep = [n for n in names if ref_of.get(n) != "R2"]
        rec = rec.channel_slice(keep)
        names = keep
        groups, _odd, ref_of = NS.tetrodes(names)
    # a tetrode's wires sit within tens of microns of each other; the exact geometry is
    # unknown, so a small square is used - enough for the sorters to treat the four as one
    # dense group and not enough to imply a real distance
    pos, grp = [], []
    idx = {n: i for i, n in enumerate(names)}
    square = [(0.0, 0.0), (25.0, 0.0), (0.0, 25.0), (25.0, 25.0)]
    for gi, (_s, _k, quad) in enumerate(groups):
        for w, n in enumerate(quad):
            if n in idx:
                pos.append((gi * 300.0 + square[w % 4][0], square[w % 4][1]))
                grp.append(gi)
    probe = pi.Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=np.asarray(pos), shapes="circle",
                       shape_params={"radius": 7.5})
    probe.set_device_channel_indices(np.arange(len(pos)))
    rec = rec.set_probe(probe)
    rec.set_property("group", np.asarray(grp))
    rec = si.bandpass_filter(rec, freq_min=BAND[0], freq_max=BAND[1])
    return rec, groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True, choices=sorted(NS.SESSIONS))
    ap.add_argument("--sorter", choices=SORTERS)
    ap.add_argument("--prepare", action="store_true", help="stage the shafts, sort nothing")
    ap.add_argument("--rasters", action="store_true", help="cue-aligned rasters per unit")
    ap.add_argument("--reref", default="shaft_r1",
                    choices=("shaft_r1", "tetrode_car", "none"))
    ap.add_argument("--drop-r2", action="store_true",
                    help="also leave out wires 5 and 9, the other reference-wired contacts")
    ap.add_argument("--files", nargs="*", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-stage even if present")
    a = ap.parse_args()

    p = a.patient
    if not (a.prepare or a.sorter or a.rasters):
        ap.error("give --prepare, --sorter or --rasters")
    if (a.sorter or a.rasters) and p not in TRIALS_OK:
        raise SystemExit(
            f"{p}: its 140 trial table does not line up with its .ns6 photodiode, so spike "
            f"times could not be placed against trials. Resolve the sync first "
            f"(microepi_03_ersp.py --patient {p} --dry-run --sync-report). "
            f"Staging with --prepare is still fine.")

    # check the toolchain BEFORE staging: staging a patient is gigabytes and many minutes
    # over the share, and failing on a missing import afterwards wastes all of it
    si = pi = None
    if a.sorter or a.rasters:
        si, pi = need_si()
        if a.sorter:
            import spikeinterface.sorters as ss
            avail = list(ss.available_sorters())
            if a.sorter not in avail:
                raise SystemExit(
                    f"{a.sorter} is not installed. SpikeInterface sees: "
                    f"{', '.join(avail) or '(none)'}\n"
                    f"    pip install mountainsort5          # for mountainsort5\n"
                    f"    pip install hdbscan numba          # for spykingcircus2\n")

    names = [n for n in NS.channel_names(p, a.files) if not NS.is_ainp(n)]
    by_shaft = {}
    for n in names:
        by_shaft.setdefault(NS.shaft_of(n), []).append(n)
    print(f"=== {p} ({NS.SESSIONS[p]['pat']}): {len(names)} microwires on "
          f"{len(by_shaft)} shafts -> outputs/{TREE}")

    staged = {}
    for shaft, wires in by_shaft.items():
        staged[shaft] = stage_shaft(p, shaft, wires, a.reref, files=a.files, force=a.force)
    if a.prepare:
        secs = sorted({round(s[3] / FS) for s in staged.values()})
        gb = sum(os.path.getsize(s[0]) for s in staged.values()) / 1e9
        span = f"{secs[0]} s" if len(secs) == 1 else f"{secs[0]}-{secs[-1]} s"
        print(f"  staged {len(staged)} shafts, {span} each at {FS/1000:.0f} kHz, "
              f"{gb:.1f} GB total")
        return 0

    if a.sorter:
        for shaft, (dat, nm, gain, n_s) in staged.items():
            rec, groups = recording_for(si, pi, dat, nm, gain, n_s, drop_r2=a.drop_r2)
            folder = out_root(p) / a.sorter / shaft
            print(f"  {shaft}: {len(groups)} tetrodes, {rec.get_num_channels()} channels "
                  f"-> {folder}")
            sorting = si.run_sorter_by_property(
                sorter_name=a.sorter, recording=rec, grouping_property="group",
                folder=str(folder), remove_existing_folder=True, verbose=True)
            n_units = len(sorting.unit_ids)
            sorting.save(folder=str(folder / "sorting"))
            print(f"  {shaft}: {n_units} units")
        return 0

    if a.rasters:
        from microepi_spike_rasters import make_rasters      # noqa: E402
        return make_rasters(p, out_root(p), staged, files=a.files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
