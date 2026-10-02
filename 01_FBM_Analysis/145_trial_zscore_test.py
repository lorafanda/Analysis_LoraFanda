#!/usr/bin/env python3
"""
145_trial_zscore_test.py - a throwaway tree for judging TRIALS, not cohorts.

    python 145_trial_zscore_test.py --patient PAT_3455
    python 145_trial_zscore_test.py --patient PAT_3390 PAT_3415 PAT_3965 PAT_3975

Writes outputs/04_ersp-trialtestzscore/<patient>/LM/ with three things and nothing else:

    HFA/<cond>/           the high-frequency per-trial rasters ("HG" until 2026-09-30)
    ERSP_clean/<cond>/    the clean ERSP images
    TrialScores/<cond>/   <patient>_<cond>_trial_scores.tsv - one row per channel x trial

IT DOES NOT FORK 140. It imports 140_ersp_pipeline and runs that module's own
process_patient, with the output roots pointed at the test tree and the product switches
turned down to those three. Every number here therefore comes from the same code that
builds the real cubes; a forked copy of an 873-line pipeline would drift from it within
a week, and then the trials you judged would not be the trials you kept.

NOTHING IN THE REAL TREE (outputs/<cfg.ERSP_TREE>, 03_ERSP since 2026-09-30) IS TOUCHED. No cubes, no halves, no PSD, no
montage, no wm_reref_report row - so this can be run on a patient whose real tree you
are happy with, and that tree is exactly as it was afterwards.

HOW THE SCORES ARE MADE, and why a threshold is set that rejects nothing.

compute_ersp already scores every trial: the 99th percentile of |dB| over that trial's
map, and again over the high-gamma band alone (trial_reject_band, 70-150 Hz). A
percentile rather than the max so one bad pixel cannot condemn a trial, and rather than
the mean so a brief loud trial is still caught. But it only computes them when some
trial_reject_* threshold is set - otherwise it returns NaN and skips the work. So this
script sets trial_reject_hg_z to an unreachable value: the scoring runs, and no trial is
ever dropped. The cubes it writes are therefore identical to a normal run's, and the
scores come out honest - scored on every trial, biased by none.

The SWEEP is a separate step, on the TSVs: 146_trial_zscore_sweep.py. Compute once,
sweep freely; nothing needs re-running to try another z.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_NAME = "04_ersp-trialtestzscore"
NEVER_REJECTS = 1e9      # a z no trial can reach: scoring on, rejection off


def load_140():
    """Import 140_ersp_pipeline as a module (its name starts with a digit)."""
    path = HERE / "140_ersp_pipeline.py"
    spec = importlib.util.spec_from_file_location("ersp140", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ersp140"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", nargs="+",
                    help="patient ids as in config.patient_ids (EL035, PAT_3455, G-01 ...)")
    ap.add_argument("--all", action="store_true",
                    help="every patient in config.patient_ids")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip a patient that already has all three score tables")
    ap.add_argument("--keep-ersp-plots", action="store_true",
                    help="also write the per-channel ERSP figures (off: HFA + clean only)")
    a = ap.parse_args()
    if not a.patient and not a.all:
        ap.error("give --patient <ids> or --all")

    m = load_140()
    if a.all:
        a.patient = [str(p) for p in m.cfg.patient_ids]

    if a.skip_done:
        keep = []
        for pid in a.patient:
            out = m.out_id_for(pid) if hasattr(m, "out_id_for") else pid
            # either layout: TrialScores/ (runs before 2026-10-02) or PerTrial/TrialScores/
            d = Path(m.cfg.outputs_root) / OUT_NAME / str(out) / m.cfg.block_name
            done = len(list(d.rglob("*_trial_scores.tsv"))) if d.is_dir() else 0
            (keep.append(pid) if done < 3
             else print(f"  skip {pid}: {done} score tables already there"))
        a.patient = keep
    root = os.path.join(m.cfg.outputs_root, OUT_NAME)

    # ---- the test tree. Both of 140's roots point INSIDE it, so every product it still
    # writes lands here and neither real tree is touched.
    m.run_root_ersp = root
    m.run_root_raw = root

    # ---- only the three products asked for
    m.WRITE_MONTAGE = False
    m.WRITE_ERSP_PLOTS = bool(a.keep_ersp_plots)
    m.WRITE_HG_PLOTS = True
    m.WRITE_SIGNAL_PLOTS = False      # the signal-trace twin (2026-10-02) is not one of this tree's products
    m.WRITE_CUBES = False
    m.WRITE_HALVES = False
    m.WRITE_CLEAN_PNG = True
    m.EXPORT_TRIAL_SCORES = True

    print(f"trial test tree : {root}")
    print(f"products        : HFA rasters, clean ERSP images, per-trial scores"
          + (", ERSP figures" if a.keep_ersp_plots else ""))
    print(f"scoring         : trial_reject_hg_z = {NEVER_REJECTS:g} "
          f"(scores computed, no trial dropped)\n")

    rc = 0
    for pid in a.patient:
        print("=" * 72)
        print(f"{pid}")
        print("=" * 72)
        try:
            # process_patient reads the thresholds off cfg at entry, so they are set on
            # cfg rather than on ersp_params - which that function would overwrite. The
            # three other arms are set to None for this patient (the patient key beats
            # the "default" key, which since 2026-09-30 carries MAD 3.5 for everyone):
            # here every trial is scored and none is dropped, whatever the real run does.
            for d, v in (("ersp_trial_reject_hg_z", NEVER_REJECTS),
                         ("ersp_trial_reject_hg_mad", None),
                         ("ersp_trial_reject", None),
                         ("ersp_trial_reject_z", None)):
                cur = dict(getattr(m.cfg, d, {}) or {})
                cur[str(pid)] = v
                setattr(m.cfg, d, cur)
            # RUN_ERSP_PIPELINE gates HG, RUN_CLUSTER_EXPORT gates the clean PNG;
            # DO_MONTAGE_PSD_PLOTS off kills the PSD montages, the slowest step here.
            row = m.process_patient(pid, True, True, False, False)
            print(f"  status: {row.get('status')}")
            if row.get("status") not in ("ok", None):
                rc = 1
        except Exception as e:                                   # noqa: BLE001
            print(f"  [error] {pid}: {type(e).__name__}: {e}")
            rc = 1
        finally:
            try:
                m.plt.close("all")
            except Exception:
                pass

    print(f"\n-> {root}")
    print("Then: python 146_trial_zscore_sweep.py   (sweeps z over the TSVs, no re-run)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
