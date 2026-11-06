"""Rerun the whole member_measures chain in order (Lora, 2026-11-03).

    python run_all.py --k 7                         every feature set at K = 7, FDR NeuroSynth maps
    python run_all.py --k 7 --ns nimare             the same with the NiMARE maps (run 0_nimare_maps.py first)
    python run_all.py --k published                 each feature set at its published K
    python run_all.py --k 7 --from 4                start at step 4 (steps before it already done)
    python run_all.py --k 7 --only 10               one step
    python run_all.py --k 7 --fset 15zband          one feature set (steps that loop over feature sets)

Steps: 1 measures, 2 white matter, 3 spatial grouping, 4 NeuroSynth sampling, 5 Bayesian ranks,
6 story figures, 7 per-cluster figures, 8 enrichment heatmaps, 9 bars, 10 label figure,
11 map composition (the seven NiMARE contrasts by default; meant for --ns nimare).
Step 2 does not depend on K and is skipped unless asked (--with-wm) or its table is missing.
Stops at the first step that fails. Each step prints what it wrote."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = {1: "1_measures.py", 2: "2_white_matter.py", 3: "3_spatial_grouping.py", 4: "4_neurosynth.py",
         5: "5_bayes_ranks.py", 6: "6_story_figures.py", 7: "7_per_cluster.py", 8: "8_neurosynth_enrichment.py",
         9: "9_neurosynth_bars.py", 10: "10_neurosynth_labels.py", 11: "11_map_composition.py"}
PER_FSET = {1, 3, 5, 6, 11}                   # steps that accept --fset

ap = argparse.ArgumentParser()
ap.add_argument("--k", default="7", help="published | a number")
ap.add_argument("--ns", default="fdr", choices=["fdr", "nimare"])
ap.add_argument("--fset", default="all")
ap.add_argument("--from", dest="start", type=int, default=1)
ap.add_argument("--only", type=int, nargs="*", default=None)
ap.add_argument("--with-wm", action="store_true", help="also rerun step 2 (white matter)")
a = ap.parse_args()

todo = a.only if a.only else [s for s in STEPS if s >= a.start]
if 2 in todo and not a.with_wm and (HERE / "out" / "white_matter.csv").exists():
    todo.remove(2)
    print("step 2 skipped: out/white_matter.csv exists and does not depend on K (--with-wm to redo)")
env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
for s in todo:
    cmd = [sys.executable, str(HERE / STEPS[s]), "--k", a.k, "--ns", a.ns]
    if s in PER_FSET and a.fset != "all":
        cmd += ["--fset", a.fset]
    print(f"\n===== step {s}: {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(HERE), env=env)
    if r.returncode:
        raise SystemExit(f"step {s} failed (exit {r.returncode}) - fix it and rerun with --from {s}")
    print(f"===== step {s} done in {time.time() - t0:.0f} s")
print("\nall done")
