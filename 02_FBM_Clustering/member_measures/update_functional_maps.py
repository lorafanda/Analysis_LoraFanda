"""Rebuild everything the Functional maps tab shows, in order, with the flags the tab names (Lora, 2026-11-07).

    python update_functional_maps.py                 steps 13-17 (screen + _poster + _paper), then the tab + the docx
    python update_functional_maps.py --tab-only      only the tab, the docx and the site UI (after a manual rerun)
    python update_functional_maps.py --no-site       everything, but do not splice into the site checkout

Run on the laptop (the site checkout is there) with the desktop unlocked (the glass-brain and pial tiles render with
VTK; existing tiles are reused). Takes about 15 minutes. Afterwards commit with pathspecs and push both repos:
    git commit --only -- <the files listed at the end>      (the analysis repo's index holds other staged files)
Maps: if 0_nimare_maps.py changed, run it and 4_neurosynth.py --ns nimare first; every step reads out/nimare_z.csv."""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ST = ["--styles", "screen", "poster", "paper"]
STEPS = [
    ["13_cluster_loadings.py", "--k", "7", "--ns", "nimare", "--fset", "15band", "--loadings", "0.35", "--v4", *ST],
    ["14_functional_information.py", "--ns", "nimare", *ST],
    ["15_cross_feature_maps.py", "--ns", "nimare", "--perm", "2000", "--v4", *ST],
    ["16_highlight_maps.py", "--ns", "nimare", "--configs", "15band:7", "15zband:7", "5band:6", "--v4", *ST],
    ["16_highlight_maps.py", "--ns", "nimare", "--configs", "15band:7", "--v4", "--all-maps", *ST],
    ["16_highlight_maps.py", "--ns", "nimare", "--configs", "15band:7", "--v4", "--maps", *ST],
    ["17_maps_by_k.py", "--ns", "nimare"],
]

ap = argparse.ArgumentParser()
ap.add_argument("--tab-only", action="store_true")
ap.add_argument("--no-site", action="store_true")
a = ap.parse_args()


def run(cmd, cwd):
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run([sys.executable, *cmd], cwd=cwd)
    if r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(cmd)}")


if not a.tab_only:
    for cmd in STEPS:
        run(cmd, HERE)
run(["make_functional_maps_tab.py", "--docx"] + ([] if a.no_site else ["--insert"]), ROOT)
if not a.no_site:
    run(["make_site_ui.py", "--insert"], ROOT)
print("""
done. Commit (analysis repo, from Analysis_LoraFanda/):
    git add -- 02_FBM_Clustering/outputs/_status_png/functional_maps
    git commit --only -- 02_FBM_Clustering/member_measures 02_FBM_Clustering/make_functional_maps_tab.py 02_FBM_Clustering/outputs/_status_png/functional_maps
and push; then in C:/Users/fanda/lorafanda.github.io commit analysis_status.html and push.""")
