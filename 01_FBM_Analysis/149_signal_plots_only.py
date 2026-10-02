#!/usr/bin/env python3
"""
149_signal_plots_only.py - TEMPORARY: the Signal/ trial figures for patients already in 03_ERSP,
without re-running the rest of 140.

    python 149_signal_plots_only.py --all                 every patient with cubes in 03_ERSP
    python 149_signal_plots_only.py --all --skip-done     ... that has no Signal/ folder yet
    python 149_signal_plots_only.py --patient PAT_3455 EL040

140 writes Signal/<cond>/<pid>_<cond>_<ref>_SIGtrials_<ch>.png itself since 2026-10-02
(WRITE_SIGNAL_PLOTS), so this script is only for the patients that were run before that. It
can go once every patient has been through 140 again.

IT DOES NOT FORK 140. Like 145, it imports 140_ersp_pipeline and runs that module's own
process_patient with every product switched off except the signal figures - so the signal
drawn is the one the pipeline cleans (same load, crop, bad-list drop, reference, notch, trial
tables and per-channel MAD rejection), from the config as it is NOW.

NOTHING ELSE IN 03_ERSP IS TOUCHED. process_patient also writes its run-level files
(Report/<pid>_IQR.tsv and the duration figures, the notch audit, the wm_reref_report row,
the log). Those describe the run that made the cubes, and must keep describing it, so both
of 140's roots are pointed at a scratch tree (outputs/_signal_only_tmp); when a patient is
done its Signal/ folder is moved into 03_ERSP/<pid>/LM/ and the scratch copy is deleted.

IF A PATIENT'S CONFIG CHANGED SINCE ITS RUN (a longer bad list, another reference), these
figures show the NEW cleaning while its cubes are still the old one. That patient needs a
real 140 run, which writes the figures anyway; the script prints the reminder.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TMP_NAME = "_signal_only_tmp"


def load_140():
    """Import 140_ersp_pipeline as a module (its name starts with a digit)."""
    spec = importlib.util.spec_from_file_location("ersp140", HERE / "140_ersp_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ersp140"] = mod
    spec.loader.exec_module(mod)
    return mod


def out_id(m, raw: str) -> str:
    """The folder name a patient is written under (MicroEPI ids run as their PAT_ name)."""
    pre = getattr(m.cfg, "MICROEPI_MAT_PRESETS", {}).get(str(raw))
    if pre:
        return pre["pat_name"]
    return str(raw) if str(raw).startswith(("EL", "PAT_")) else f"PAT_{raw}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", nargs="+", help="ids as in config.patient_ids (EL035, PAT_3455, G-01 ...)")
    ap.add_argument("--all", action="store_true", help="every patient of config.patient_ids with cubes in the tree")
    ap.add_argument("--skip-done", action="store_true", help="skip a patient whose Signal/ folder already has figures")
    a = ap.parse_args()
    if not a.patient and not a.all:
        ap.error("give --patient <ids> or --all")

    m = load_140()
    cfg = m.cfg
    tree = Path(cfg.outputs_root) / cfg.ERSP_TREE
    tmp = Path(cfg.outputs_root) / TMP_NAME
    sdir = getattr(cfg, "SIGNAL_DIR", "Signal")

    ids = [str(p) for p in (cfg.patient_ids if a.all else a.patient)]
    todo = []
    for raw in ids:
        pid = out_id(m, raw)
        if not (tree / pid / cfg.block_name / "ERSP_matrix").is_dir():
            print(f"  skip {raw}: no cubes in {tree.name} yet - its 140 run will write the figures")
            continue
        done = len(list((tree / pid / cfg.block_name / sdir).glob("*/*.png")))
        if a.skip_done and done:
            print(f"  skip {raw}: {done} signal figures already there")
            continue
        todo.append((raw, pid))
    if not todo:
        print("nothing to do")
        return 0

    # both roots into the scratch tree; only the signal figures are a product
    m.run_root_ersp = str(tmp)
    m.run_root_raw = str(tmp)
    m.WRITE_MONTAGE = False
    m.WRITE_ERSP_PLOTS = False
    m.WRITE_HG_PLOTS = False
    m.WRITE_SIGNAL_PLOTS = True
    m.WRITE_CUBES = False
    m.WRITE_HALVES = False
    m.WRITE_CLEAN_PNG = False
    m.EXPORT_TRIAL_SCORES = False

    print(f"tree     : {tree}")
    print(f"scratch  : {tmp}   (run-level files land here and are deleted)")
    print(f"scale    : {getattr(cfg, 'signal_plot_uv_per_row', 300.0):g} µV per row, "
          f"{getattr(cfg, 'signal_plot_rows', 58)} rows, every patient")
    print(f"patients : {' '.join(r for r, _ in todo)}\n")

    rc = 0
    for raw, pid in todo:
        print("=" * 72 + f"\n{raw}" + (f" ({pid})" if pid != raw else "") + "\n" + "=" * 72)
        try:
            # RUN_ERSP_PIPELINE on (the figure lives in that branch), cluster export off,
            # PSD montages off, micros off
            row = m.process_patient(raw, True, False, False, False)
            status = row.get("status")
        except Exception as e:                       # one patient must not stop the others
            print(f"  [error] {raw}: {type(e).__name__}: {e}")
            rc = 1
            continue
        src = tmp / pid / cfg.block_name / sdir
        n = len(list(src.glob("*/*.png"))) if src.is_dir() else 0
        if not n:
            print(f"  [warn] {raw}: no signal figure written (status {status})")
            rc = 1
        else:
            dst = tree / pid / cfg.block_name / sdir
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            print(f"  {n} figures -> {dst}")
        shutil.rmtree(tmp / pid, ignore_errors=True)
    for leftover in ("logs", "wm_reref_report.tsv"):
        p = tmp / leftover
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
    try:
        tmp.rmdir()
    except OSError:
        pass
    print("\nreminder: a patient whose bad list or reference changed since its 03_ERSP run now has "
          "signal figures of the NEW cleaning beside cubes of the old one - re-run 140 for it.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
