#!/usr/bin/env python
"""140 with another baseline window, written to a TEST tree - nothing in outputs/03_ERSP changes.

    python .\140_baseline_test.py --patient PAT_1327
    python .\140_baseline_test.py --patient PAT_1327 --baseline -0.8 -0.1 --out outputs/_baseline_test
    python .\140_baseline_test.py --patient PAT_1327 --no-psd          (anything after the patient goes to 140)

THE QUESTION (2026-10-15, Lora): is the response-locked rise seen on all channels of some
patients made by the baseline? The ERSP is dB re the per-trial baseline mean over
cfg.baseline_calc_w (-0.4 .. -0.1 s before the onset; cfg.baseline_w, -0.6 .. -0.1, is the
window the rasters and signal figures mark). This runs the same pipeline with both set to the
window given (default -0.8 .. -0.1 s, twice as long) and the whole output tree moved to --out,
so the cubes, figures, report and logs of the real tree stay untouched and 141 / the cache never
see the test. If the rise is still there, it is the response period, not the baseline.

HOW. The config module is shared with 140 (functions.config); its two baseline tuples and
outputs_root are overridden here before 140 is executed in this process, so 140 builds
ersp_params and its output roots from the test values. Run it with the same Python as 140
(mne, h5py, scipy).
"""
import argparse
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
from functions import config as cfg          # noqa: E402  (the same module object 140 imports)

ap = argparse.ArgumentParser()
ap.add_argument("--patient", required=True, nargs="+")
ap.add_argument("--baseline", type=float, nargs=2, default=(-0.8, -0.1), metavar=("LO", "HI"),
                help="baseline window in s relative to the onset (default -0.8 -0.1)")
ap.add_argument("--out", default=os.path.join("outputs", "_baseline_test"),
                help="test output root; the tree is <out>/<cfg.ERSP_TREE>/<pid>/LM/ (default outputs/_baseline_test)")
a, passthrough = ap.parse_known_args()

lo, hi = float(a.baseline[0]), float(a.baseline[1])
assert lo < hi <= 0, "the baseline window must end at or before the onset"
assert lo >= float(cfg.time_window[0]), f"the baseline starts before the epoch ({cfg.time_window[0]} s)"
out_root = os.path.abspath(a.out)
os.makedirs(out_root, exist_ok=True)

print(f"[baseline test] baseline_w {cfg.baseline_w} -> ({lo}, {hi}); baseline_calc_w {cfg.baseline_calc_w} -> ({lo}, {hi})")
print(f"[baseline test] outputs_root -> {out_root}   (the real tree {cfg.outputs_root} is not written)")
cfg.baseline_w = (lo, hi)
cfg.baseline_calc_w = (lo, hi)
cfg.outputs_root = out_root

sys.argv = ["140_ersp_pipeline.py", "--patient", *a.patient, *passthrough]
runpy.run_path(os.path.join(HERE, "140_ersp_pipeline.py"), run_name="__main__")
