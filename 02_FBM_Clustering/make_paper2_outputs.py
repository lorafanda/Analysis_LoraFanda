#!/usr/bin/env python3
"""
make_paper2_outputs.py - everything Paper 2 shows, rebuilt on the current cohort, in one
command.

    python make_paper2_outputs.py                 all of it (needs the screen awake)
    python make_paper2_outputs.py --headless      skip the five steps that render brains
    python make_paper2_outputs.py --list          print the plan and stop
    python make_paper2_outputs.py --only bundles  one phase (see PHASES below)

WHAT IT DOES NOT DO, and why.

  249's statistics figures.  make_cluster_statistics.py, make_cluster_figures.py and
  make_heldout_figure.py are NOT here. 249 itself still has to run - the peak K per
  feature set comes from its held-out sweep and the page's numbers come from its
  statistics - but its C3 / C8 / C13 figures are not part of this rebuild. Run 249
  first, separately, when the cohort changes.

  The cohort.  rebuild_concat_cache.py and notebooks 240 / 241 / 242 come before all of
  this. This script reads runs; it does not fit any.

  252's glassbrains.  A notebook, and a long one.

HOW LONG. About two hours on cohort v10, and one step is most of it: the first
make_missing_centroids.py --force on a fresh set of runs took 68 minutes (it draws every
cluster of every K of all twelve runs). Once those exist, the rest is roughly 25 minutes.
--only paper / --only site are the short loops to iterate in.

THE SCREEN. Five steps render brains through pyvista, which needs a GL context: FIG 0,
FIG 1, both FIG 2 passes and FIG 3. On a locked screen VTK cannot get one and the process
dies with an access violation on wglChoosePixelFormatARB, no traceback. Run them at the
desk, or pass --headless and run them later.

THE PATH SPELLING. prepare_dataset keys its cache on the input_dir it was built with, so
a script reached through S:\\ is handed a cache built through \\\\nasac-m2... and refuses it.
Every step here is launched from the nasac spelling whatever spelling this file was
reached by, which is why the run directories in the log all start \\\\nasac-m2.

Nothing aborts the run: every step is logged with its exit code and the next one starts,
so one broken generator does not cost the other thirty. The summary at the end is the
report, and audit_site_figures.py has the last word on what is still old.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# the repo, always through the nasac spelling (see THE PATH SPELLING above)
NASAC = Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\ANALYSIS\FLM\Analysis_LoraFanda")
TAIL = ("ANALYSIS", "FLM", "Analysis_LoraFanda")
_here = Path(__file__).resolve().parents[1]
ROOT = NASAC if tuple(_here.parts[-3:]) == TAIL else _here
CLUST, PREP = ROOT / "02_FBM_Clustering", ROOT / "01_FBM_Analysis"
KS = ["6", "7", "8", "9", "10"]                       # the K range the page shows FIG 4 / 5 at


def newest_cnmf_hg() -> str:
    d = CLUST / "outputs" / "clustering" / "cnmf" / "concat_hg" / "runs"
    runs = sorted(x.name for x in d.iterdir() if x.is_dir())
    return f"cnmf/concat_hg/runs/{runs[-1]}"


# (phase, needs a screen, cwd, argv)
def plan() -> list[tuple[str, bool, Path, list[str]]]:
    return [
        # per-run products: the gallery and the visualizer both read these, and a fresh
        # run has neither until they are written
        ("runs", False, CLUST, ["make_missing_centroids.py", "--force"]),
        ("runs", False, CLUST, ["make_centroid_rasters.py"]),

        # the paper figures that render brains
        ("paper", True, CLUST, ["00_paper2_figure0_coverage.py"]),
        ("paper", True, CLUST, ["00_Paper2_Figures.py"]),
        ("paper", True, CLUST, ["00_paper2_figures2_2.py"]),
        ("paper", True, CLUST, ["00_paper2_figures2_2.py", "--algo-feature-set", "concat_bands5z"]),
        ("paper", True, CLUST, ["00_paper2_figure3_lana.py"]),

        # and the ones that do not
        *[("paper", False, CLUST, ["00_paper2_figure4_correspondence.py", "--k", k]) for k in KS],
        *[("paper", False, CLUST, ["00_paper2_figure4_panels.py", "--k", k]) for k in KS],
        ("paper", False, CLUST, ["00_paper2_figure5_confusion.py", "--k", *KS]),

        # the stage-02 figures the status page carries
        ("stage02", False, CLUST, ["make_separation_figure.py"]),
        ("stage02", False, CLUST, ["make_separation_across_features.py"]),
        ("stage02", False, CLUST, ["make_decomposition_figure.py"]),
        ("stage02", False, CLUST, ["make_overview_options.py"]),
        ("stage02", False, CLUST, ["make_component_anatomy.py", "--run", newest_cnmf_hg()]),
        ("stage02", False, CLUST, ["make_archetype_explainer.py"]),      # E10
        ("stage02", False, CLUST, ["make_membership_explainer.py"]),     # E12
        # FIG C.9 and G2 read a run of the UNGATED set (concat_hg_all). Notebook 237 is
        # not part of the standard rebuild, so this step fails until 237 has run on the
        # current cohort - deliberately left in, so the summary says so out loud.
        ("stage02", False, CLUST, ["make_gate_split_figures.py"]),

        # normalisation, and the preprocessing side
        ("norm", False, CLUST, ["make_normalisation_figures.py"]),
        ("norm", False, CLUST, ["make_norm_notes.py", "--insert"]),
        ("prep", False, PREP, ["141_audit_140.py"]),
        ("prep", False, PREP, ["make_preprocessing_figures.py"]),
        ("prep", False, PREP, ["make_s1_tab.py", "--insert"]),

        # coverage FIRST: it writes the manifest the other two resolve runs through
        ("bundles", False, CLUST, ["make_coverage_bundle.py"]),
        ("bundles", False, CLUST, ["make_centroid_bundle.py"]),
        ("bundles", False, CLUST, ["make_cluster_visualizer.py"]),

        # the page, then the audit that says what is still old
        ("site", False, CLUST, ["make_paper_figures_webblock.py", "--insert"]),
        ("site", False, CLUST, ["make_cluster_webblock.py", "--insert"]),
        ("site", False, CLUST, ["make_s2_gallery.py", "--insert"]),
        ("site", False, CLUST, ["make_site_ui.py", "--insert"]),
        ("site", False, CLUST, ["audit_site_figures.py", "--csv"]),
    ]


PHASES = ["runs", "paper", "stage02", "norm", "prep", "bundles", "site"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true",
                    help="skip the five pyvista steps (FIG 0, 1, 2 x2, 3)")
    ap.add_argument("--only", nargs="*", choices=PHASES, help="run only these phases")
    ap.add_argument("--list", action="store_true", help="print the plan and stop")
    a = ap.parse_args()

    steps = [s for s in plan()
             if (not a.only or s[0] in a.only) and not (a.headless and s[1])]
    print(f"{ROOT}\n{len(steps)} steps"
          + (" (headless: 5 brain renders skipped)" if a.headless else "")
          + f"\n{'-' * 72}", flush=True)
    if a.list:
        for ph, gl, cwd, argv in steps:
            print(f"  {ph:<8} {'GL ' if gl else '   '} {cwd.name:<18} {' '.join(argv)}")
        return 0

    results = []
    for ph, gl, cwd, argv in steps:
        print(f"\n[{datetime.now():%H:%M:%S}] {ph} $ python " + " ".join(argv), flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, *argv], cwd=str(cwd),
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        results.append((ph, " ".join(argv), r.returncode, time.time() - t0))
        print(f"[{datetime.now():%H:%M:%S}]    exit {r.returncode} in {time.time() - t0:.0f}s",
              flush=True)

    print(f"\n{'=' * 72}\nSUMMARY", flush=True)
    for ph, name, rc, secs in results:
        print(f"{'ok  ' if rc == 0 else 'FAIL'}  {ph:<8} {name:<62} {secs:>6.0f}s")
    bad = [n for _, n, rc, _ in results if rc]
    print(f"\n{len(results) - len(bad)}/{len(results)} ok"
          + ("" if not bad else "\nFAILED:\n  " + "\n  ".join(bad)))
    print("\nCommit what changed, then re-run make_s2_gallery.py --insert and\n"
          "make_paper_figures_webblock.py --insert: both refuse to show a figure that is\n"
          "not committed, because the page fetches them from raw.githubusercontent.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
