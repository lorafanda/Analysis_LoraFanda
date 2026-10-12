#!/usr/bin/env python3
"""microepi_03_ersp.py - 03_ERSP, but on the MicroEPI microwires from the 30 kHz .ns6.

    python microepi_03_ersp.py --patient G-03
    python microepi_03_ersp.py --all
    python microepi_03_ersp.py --patient G-05 --reref tetrode_car

WHAT IT IS. 140_ersp_pipeline.py is imported, not copied: the ERSP maths, the notch, the
trial rejection and every product below ERSP/ ERSP_clean/ ERSP_matrix/ PerTrial/ are the
ones 03_ERSP already uses, so a micro cube and a macro cube are the same object computed
the same way. Three things are redirected -

  1. the output roots, to outputs/microepi_03_ersp/<PID>/LM/...
  2. the loader, to the .ns6 micro channels instead of the .mat macros
  3. collect_trials, to put 140's own trials on the .ns6 clock

- and nothing else is touched. Nothing in DATARAW is written.

WHY THE TRIALS MOVE. 140's trials live in the export's clock at 2048 Hz; the micros live
in the Blackrock clock at 30 kHz, and the two run about 17.5 ppm apart and disagree by
8-10 ms per trial on where the same screen flash began. Each trial is therefore matched to
ITS OWN photodiode pulse in the .ns6 rather than mapped through a fitted line, so the
epochs sit correctly against the micro samples. 140's curation (IQR, bad_spans,
min_stim_s, max_post_s) is applied first and carried over intact: only the times move, the
trial set and the epoch geometry do not.

RATE. The micros are read at 30 kHz and decimated to 1 kHz with an anti-aliased polyphase
filter before the ERSP. The ERSP reaches 400 Hz, so 1 kHz is enough for it, and 48
channels of 30 kHz is about 5 GB of int16 per patient before anything is cast to float.
The full rate is kept for spike sorting, which is a separate branch.

REFERENCE. Wire 1 of each shaft is R1, wired as a hardware reference, and is subtracted
from the rest of its shaft (--reref, see lf_microepi_ns6.reref). 140's own WM/CAR
re-referencing is left inert: it keys off white-matter contacts in the electrodes TSV,
which the microwires are not in, and the micro reference has to stay local to the shaft.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "functions"))

import lf_microepi_ns6 as NS                      # noqa: E402
from functions import config as cfg               # noqa: E402

TREE = "microepi_03_ersp"
ERSP_FS = 1000.0            # what the micros are decimated to before the ERSP
DEFAULT_REREF = "shaft_r1"
_ORIG_COLLECT = None        # lf_trials.collect_trials as it was before any patching


def load_140():
    """140 as a module. Its argparse sits under __main__, so importing it is side-effect
    free apart from the module-level roots, which are overwritten below."""
    path = HERE / "140_ersp_pipeline.py"
    spec = importlib.util.spec_from_file_location("p140", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["p140"] = m
    spec.loader.exec_module(m)
    return m


def export_fs(trials):
    """The export clock, from the TSV itself: `sample` divided by `onset` in seconds.
    Read rather than assumed - it is 2048 Hz on every MicroEPI patient seen so far, but a
    wrong value here would silently rescale every duration filter in collect_trials."""
    r = [t["sample"] / t["onset"] for t in trials
         if t.get("sample", -1) > 0 and t["onset"] > 1e-6]
    return float(np.round(np.median(r))) if r else 2048.0


def install(m, patient, reref_scheme, ersp_fs, files=None, verbose=True, report=False):
    """Point 140 at the .ns6 micros for this patient. Returns the trial record."""
    root = os.path.join(cfg.outputs_root, TREE)
    m.run_root_ersp = root
    m.run_root_raw = root

    trials, sync = NS.trials_on_ns6(patient, files=files, verbose=verbose, report=report)
    fs_exp = export_fs(trials)
    located = {round(t["onset"], 4): t for t in trials if t.get("t_on") is not None}
    if verbose:
        print(f"  export clock {fs_exp:.0f} Hz, micro clock {ersp_fs:.0f} Hz after decimation")

    state = {}

    def loader(pid_raw):
        d = NS.read_ns6(patient, files=files,
                        want=[n for n in _all_micro_names(patient, files)],
                        decimate_to=ersp_fs, verbose=verbose)
        x, names, dropped = NS.reref(d["x"], d["names"], scheme=reref_scheme)
        if verbose and dropped:
            print(f"  reref {reref_scheme}: dropped {len(dropped)} reference wires "
                  f"({', '.join(dropped)})")
            print(f"  {x.shape[1]} micro channels into the ERSP")
        state["t0"] = float(d["t"][0])
        state["fs"] = float(d["fs"])
        prep_dir = _prep_dir_for(patient)
        # electrodes_tsv is None on purpose: with no WM contacts to find, 140's own
        # re-referencing stays inert and the shaft reference above is the only one applied
        return (NS.SESSIONS[patient]["pat"], x, names, float(d["fs"]), prep_dir, None,
                np.ones(x.shape[1], dtype=bool))

    # lf_trials is a SHARED module: patching m.tr.collect_trials outlives load_140(), so
    # taking the current value here would wrap the previous patient's wrapper (and use its
    # trial table). With --all that chained one wrapper per patient and every trial after
    # the first patient failed to match. Always wrap the pristine function.
    global _ORIG_COLLECT
    if _ORIG_COLLECT is None:
        _ORIG_COLLECT = m.tr.collect_trials
    orig_collect = _ORIG_COLLECT

    def collect(prep_dir, fs_hz, **kw):
        """140's trials, relocated onto the .ns6 clock.

        orig_collect reads the `sample` column straight out of the TSV, so it returns
        EXPORT samples whatever fs is passed; fs only drives its duration filters, which
        is why the export rate goes in here and the micro rate is applied afterwards.
        """
        groups = orig_collect(prep_dir, fs_exp, **kw)
        out, lost = {}, 0
        for cond, (on, off, tend) in groups.items():
            on = np.asarray(on, dtype="float64")
            off = np.asarray(off, dtype="float64")
            tend = np.asarray(tend, dtype="float64")
            n_on, n_off, n_end = [], [], []
            for k in range(len(on)):
                t_exp = on[k] / fs_exp
                rec = located.get(round(t_exp, 4))
                if rec is None:                      # tolerate rounding in the TSV
                    near = [v for key, v in located.items() if abs(key - t_exp) < 0.01]
                    rec = near[0] if near else None
                if rec is None:
                    lost += 1
                    continue
                t_on = rec["t_on"] - state.get("t0", 0.0)
                # keep 140's geometry: the stimulus and post-stimulus lengths are its own,
                # only the anchor moves to where this trial's pulse is in the .ns6
                d_off = (off[k] - on[k]) / fs_exp
                d_end = (tend[k] - on[k]) / fs_exp
                f = state.get("fs", ersp_fs)
                n_on.append(int(round(t_on * f)))
                n_off.append(int(round((t_on + d_off) * f)))
                n_end.append(int(round((t_on + d_end) * f)))
            if n_on:
                out[cond] = (np.asarray(n_on), np.asarray(n_off), np.asarray(n_end))
        if verbose:
            kept = sum(len(v[0]) for v in out.values())
            print(f"  trials on the .ns6 clock: {kept} kept"
                  + (f", {lost} with no matching pulse" if lost else ""))
        return out

    m._load_signals_and_prep_for_patient = loader
    m.tr.collect_trials = collect
    # the dry run exercises these without reading the micro samples
    state.setdefault("t0", 0.0)
    state.setdefault("fs", ersp_fs)
    return trials, sync, collect, state


def _prep_dir_for(patient):
    """Where the triggerPD tsvs are, for collect_trials to read."""
    import glob as _g
    pat = NS.SESSIONS[patient]["pat"]
    pr = getattr(cfg, "MICROEPI_MAT_PRESETS", {}).get(patient)
    if pr:
        return os.path.join(os.path.dirname(str(pr["data_dir"])), "prep0")
    ov = getattr(cfg, "PATH_OVERRIDES", {}).get(pat, {})
    for c in _g.glob(os.path.join(str(ov.get("base_root", "")), "*", "prep0")):
        if _g.glob(os.path.join(c, f"{pat}_*triggerPD*.tsv")):
            return c
    raise SystemExit(f"{patient}: no prep0 found")


def _all_micro_names(patient, files=None):
    """Every microwire, from the header alone."""
    return [n for n in NS.channel_names(patient, files) if not NS.is_ainp(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", choices=sorted(NS.SESSIONS))
    ap.add_argument("--all", action="store_true", help="every patient with microwires")
    ap.add_argument("--reref", default=DEFAULT_REREF,
                    choices=("shaft_r1", "tetrode_car", "none"))
    ap.add_argument("--ersp-fs", type=float, default=ERSP_FS)
    ap.add_argument("--files", nargs="*", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve trials and channels, write nothing")
    ap.add_argument("--sync-report", action="store_true",
                    help="with --dry-run: print every trial's residual against the first "
                         "clock segment, to see where the export and the .ns6 part company")
    a = ap.parse_args()
    if not a.patient and not a.all:
        ap.error("give --patient or --all")
    pats = sorted(NS.SESSIONS) if a.all else [a.patient]

    rc = 0
    for p in pats:
        print(f"\n=== {p} ({NS.SESSIONS[p]['pat']}) -> outputs/{TREE}")
        m = load_140()
        try:
            trials, sync, collect, state = install(m, p, a.reref, a.ersp_fs,
                                                   files=a.files, report=a.sync_report)
        except SystemExit as e:
            print(f"[skip] {p}: {e}")
            rc = 1
            continue
        if a.dry_run:
            # exercise the two things that actually break - the channel/reference layout
            # and the trial relocation - without reading a single micro sample
            names = _all_micro_names(p, a.files)
            groups, odd, ref_of = NS.tetrodes(names)
            fake = np.zeros((2, len(names)), dtype="float32")
            _x, kept, dropped = NS.reref(fake, names, scheme=a.reref)
            print(f"  {len(names)} microwires, {len(groups)} tetrodes"
                  + (f", {len(odd)} wires outside a tetrode: {odd}" if odd else ""))
            print(f"  reref {a.reref}: {len(kept)} channels kept"
                  + (f", dropped {', '.join(dropped)}" if dropped else ""))
            r2 = sorted(n for n, k in ref_of.items() if k == "R2")
            print(f"  R2 wires still in the data ({len(r2)}): {', '.join(r2)}")
            try:
                cg = collect(_prep_dir_for(p), a.ersp_fs)
                tot = sum(len(v[0]) for v in cg.values())
                if not tot:
                    print("  [fail] no epoch survived the relocation")
                    rc = 1
                    continue
                per = ", ".join(f"{c} {len(v[0])}" for c, v in sorted(cg.items()))
                last = max(int(v[2].max()) for v in cg.values() if len(v[2]))
                print(f"  epochs on the micro clock: {tot} ({per}); "
                      f"last sample {last} = {last/a.ersp_fs:.0f} s")
            except Exception as e:
                print(f"  [fail] trial relocation: {type(e).__name__}: {e}")
                rc = 1
            continue
        m.process_patient(p, RUN_ERSP_PIPELINE=True, RUN_CLUSTER_EXPORT=True,
                          DO_MONTAGE_PSD_PLOTS=True, INCLUDE_MICROEPI_MICROS=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
