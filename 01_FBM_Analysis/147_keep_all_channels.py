#!/usr/bin/env python3
"""147_keep_all_channels.py - 140, but nothing is dropped for being bad-listed.

    python 147_keep_all_channels.py --patients PAT_3455 PAT_3965 PAT_3975
    python 147_keep_all_channels.py --patients PAT_3455 --bad-in-ref

WHY. 140 writes NOTHING for a bad-listed contact: no cube, no ERSP, no raster. So a
removed contact cannot be looked at, only trusted. This runs the same pipeline with the
bad list switched off for the products, into a tree of its own, so the contacts that were
thrown away can be compared against the ones that were kept.

TWO QUESTIONS, TWO TREES.
  (default)      outputs/03_ERSP-allchan
                 the bad contacts are PLOTTED but still kept OUT of the white-matter
                 mean. Every good channel is therefore identical to its 03_ERSP version,
                 so this is a clean A/B: only the extra contacts are new.
  --bad-in-ref   outputs/03_ERSP-allchan-ref
                 the bad contacts go back INTO the WM mean as well. Now the reference
                 changes, so EVERY channel changes and nothing is comparable to 03_ERSP -
                 but the difference between the two trees on a GOOD channel is exactly
                 what those contacts were contributing to the reference. That is the
                 measurement worth having on PAT_3455, whose reference is built from only
                 21 contacts and whose channels share 43% of their energy.

NOTHING ELSE MOVES. config.py is not edited; the bad list is restored on the way out, and
03_ERSP is never written to. 140 is imported, as 145 does, not copied.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load_140():
    spec = importlib.util.spec_from_file_location("p140", HERE / "140_ersp_pipeline.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["p140"] = m
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="+", required=True)
    ap.add_argument("--bad-in-ref", action="store_true",
                    help="also put the bad contacts back into the white-matter mean")
    ap.add_argument("--keep-wm", action="store_true",
                    help="also write products for the contacts that FORM the WM mean. "
                         "140 skips them (skip |= wm_skip), so a reference contact has no "
                         "cube, no ERSP and no raster anywhere - 93 of PAT_3455's 116 "
                         "channels survive precisely because 21 are in the reference and "
                         "2 are bad-listed. They still form the mean; they are only "
                         "no longer hidden from the products.")
    ap.add_argument("--fast", action="store_true",
                    help="write only what a channel review needs: the per-trial Signal "
                         "and HFA rasters and the trial scores. The ERSP figures, the "
                         "clean PNGs, the cubes, the halves and the montage are the slow "
                         "products and none of them is read here - the cubes are dB re "
                         "baseline, which divides out exactly the uniform noise a review "
                         "is looking for.")
    ap.add_argument("--tree", default=None, help="override the output tree name")
    a = ap.parse_args()

    m = load_140()
    cfg = m.cfg
    tree = a.tree or ("03_ERSP-allchan"
                      + ("-ref" if a.bad_in_ref else "")
                      + ("-wm" if a.keep_wm else ""))
    root = os.path.join(cfg.outputs_root, tree)
    m.run_root_ersp = root
    m.run_root_raw = root

    if a.fast:
        for off in ("WRITE_MONTAGE", "WRITE_ERSP_PLOTS", "WRITE_CUBES",
                    "WRITE_HALVES", "WRITE_CLEAN_PNG"):
            setattr(m, off, False)
        for on in ("WRITE_HG_PLOTS", "WRITE_SIGNAL_PLOTS", "EXPORT_TRIAL_SCORES"):
            setattr(m, on, True)
        print("fast: per-trial Signal and HFA rasters and trial scores only")

    saved = copy.deepcopy(getattr(cfg, "bad_channels_manual", {}))
    state = {"pid": None}

    if not a.bad_in_ref:
        # the bad list is about to be emptied so that the products are written for every
        # contact; the reference must not notice. apply_wm_reference_with_exclusions is
        # where 140 feeds it in, so it is given the saved list instead of the empty one.
        orig_reref = m.fe.apply_wm_reference_with_exclusions

        def reref(signals, names, wm_idx, bad):
            return orig_reref(signals, names, wm_idx, saved.get(state["pid"], []))
        m.fe.apply_wm_reference_with_exclusions = reref

    if a.keep_wm:
        # apply_wm_reref returns (signals, label, wm_skip) and 140 does `skip |= wm_skip`,
        # which is what hides the reference contacts. The mean is already applied by then,
        # so emptying the third element changes nothing about the referencing - only
        # whether those contacts get drawn.
        orig_wm = m.apply_wm_reref

        def keep(*args, **kw):
            sig, lab, _skip = orig_wm(*args, **kw)
            return sig, lab, set()
        m.apply_wm_reref = keep

    orig_loader = m._load_signals_and_prep_for_patient

    def loader(pid_raw):
        out = orig_loader(pid_raw)
        state["pid"] = out[0]            # the resolved patient id, for the reference above
        return out
    m._load_signals_and_prep_for_patient = loader

    cfg.bad_channels_manual = {}
    where = "and back in the WM mean" if a.bad_in_ref else "but still out of the WM mean"
    print(f"bad-listed contacts are kept {where}\n-> outputs/{tree}\n")
    try:
        for pid in a.patients:
            n = len(saved.get(pid, []) or [])
            print(f"=== {pid}: {n} bad-listed contacts come back")
            m.process_patient(pid, RUN_ERSP_PIPELINE=True, RUN_CLUSTER_EXPORT=True,
                              DO_MONTAGE_PSD_PLOTS=True, INCLUDE_MICROEPI_MICROS=True)
    finally:
        cfg.bad_channels_manual = saved      # never leave the config altered in-process
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
