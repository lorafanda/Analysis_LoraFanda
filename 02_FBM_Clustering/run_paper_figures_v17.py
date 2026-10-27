"""Every paper-2 figure on cohort v17, at the K the dominance rule chose.

WHY NOT make_paper2_outputs.py --only paper: its plan is stale for v17 in three ways.
It calls 00_Paper2_Figures.py with no --k, so the figures are drawn at the HELD-OUT PEAK
(k_peak in heldout_peaks_cnmf.csv: 30 for 15 bands, 29 for 2 bands) - the K your own rule
rejects. It calls figure0_coverage and figure3_lana with no --feature-set, so both fall
back to their default concat_hg. And it names concat_bands5z explicitly. The last two are
retired sets with no v17 run.

K PER FEATURE SET, from 249 on v17 (largest K with no cluster over half one patient):

    concat_bands15   7        concat_bands2   10   <- isolated pass, K=9 fails
    concat_bands5    8        concat_hfa       6

MAIN is the set the single-set figures are drawn for (FIG 0 coverage, FIG 2, FIG 3). It is
a judgement call, not something the data decides, so it is a constant here rather than
buried: change it and K_MAIN together.

BRAINS NEED THE SCREEN AWAKE. FIG 0, 1, 2 and 3 render through PyVista/VTK and segfault on
a locked screen (wglChoosePixelFormatARB). --headless skips exactly those.

FIG 5 must be ONE process with every K, because it also writes FIG5_pairs_allK and
FIG5_KbyK aggregated over whatever it was given. FIG 4 is split on purpose: correspondence
does the permutations and writes the tables, panels redraws from them in seconds.

NOTHING IS PUSHED. The figures land in outputs/clustering/paper_figures/.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# from __file__, so this runs on the laptop (S:) and on the server, where the repo is
# reached as a UNC path.
CLUST = Path(__file__).resolve().parent

K = {"concat_bands15": 7, "concat_bands5": 8, "concat_bands2": 10, "concat_hfa": 6}
MAIN, K_MAIN = "concat_bands15", 7          # the set FIG 0 / 2 / 3 are drawn for
FIG4_KS = [7, 8, 10]                        # K values to compare across runs
FIG5_KS = [5, 6, 7, 8, 9, 10]


def plan(headless):
    steps = []
    # --- FIG 0: coverage. Its default feature set is concat_hg, which is retired.
    steps.append((True, ["00_paper2_figure0_coverage.py",
                         "--feature-set", MAIN, "--k", str(K_MAIN)]))
    # --- FIG 1: one call per set, each at ITS OWN K. --k applies to every feature set in
    #     one call, so four calls is the only way to give four different K values.
    for fs, k in K.items():
        steps.append((True, ["00_Paper2_Figures.py",
                             "--feature-set", fs, "--k", str(k)]))
    # --- FIG 2: agreement across algorithms, on one feature set
    steps.append((True, ["00_paper2_figures2_2.py",
                         "--k", str(K_MAIN), "--algo-feature-set", MAIN]))
    # --- FIG 3: Lana, on one feature set
    steps.append((True, ["00_paper2_figure3_lana.py",
                         "--feature-set", MAIN, "--k", str(K_MAIN)]))
    # --- FIG 4: permutations first, then the fast redraw. Safe in parallel per K, but
    #     kept serial here so a failure is easy to read.
    for k in FIG4_KS:
        steps.append((False, ["00_paper2_figure4_correspondence.py", "--k", str(k)]))
    for k in FIG4_KS:
        steps.append((False, ["00_paper2_figure4_panels.py", "--k", str(k)]))
    # --- FIG 5: ONE process, all K
    steps.append((False, ["00_paper2_figure5_confusion.py", "--k", *[str(k) for k in FIG5_KS]]))
    # --- FIG 3b: the LanA spin null and the network's own response (2026-10-26). Last,
    #     so the step numbers above do not move; no brains, so it runs headless.
    steps.append((False, ["00_paper2_figure3_spin.py",
                          "--feature-set", MAIN, "--k", str(K_MAIN)]))
    return [(brains, cmd) for brains, cmd in steps if not (headless and brains)]


def paper_preview_pdf():
    """Every paper PNG at its printed width, 174 mm, on an A4 page: the renders are
    judged at the size they will be read at, not at screen size, where the printed
    font sizes look large (2026-10-27). About 2 MB; not for git."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    pdir = CLUST / "outputs" / "clustering" / "paper_figures" / "paper"
    pngs = sorted(pdir.glob("FIG*.png"))
    if not pngs:
        return None
    out = pdir / "paper_preview_A4.pdf"
    W, H, w_in = 8.27, 11.69, 174 / 25.4
    with PdfPages(out) as pdf:
        for p in pngs:
            with Image.open(p) as im:
                h_in = w_in * im.height / im.width
                fig = plt.figure(figsize=(W, H))
                ax = fig.add_axes([(W - w_in) / 2 / W, 1 - (0.7 + h_in) / H, w_in / W, h_in / H])
                ax.imshow(im)
                ax.axis("off")
                fig.text(0.5, 0.025, f"{p.name}   ·   174 mm wide   ·   {im.width} x {im.height} px",
                         ha="center", fontsize=7, color="#777")
                pdf.savefig(fig, dpi=150)
                plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true",
                    help="skip FIG 0/1/2/3, the ones that render brains")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    ap.add_argument("--only", type=int, nargs="*",
                    help="run only these step numbers (see --dry-run)")
    ap.add_argument("--paper", action="store_true",
                    help="draw every figure for the page (PAPER2_PAPER=1 for each script: "
                         "174 mm, printed font sizes, no prose or stamp, PNGs under "
                         "paper_figures/paper/); the working renders are untouched")
    a = ap.parse_args()

    steps = plan(a.headless)
    print(f"{len(steps)} steps, K = {K}, MAIN = {MAIN} (K={K_MAIN})"
          + ("   \u00b7   PAPER MODE: paper_figures/paper/" if a.paper else "") + "\n")
    for i, (brains, cmd) in enumerate(steps, 1):
        print(f"  {i:2d}. {'[brains] ' if brains else '          '}{' '.join(cmd)}")
    if a.dry_run:
        return 0
    print()

    fails = []
    for i, (brains, cmd) in enumerate(steps, 1):
        if a.only and i not in a.only:
            continue
        print(f"\n===== {i}/{len(steps)}  {' '.join(cmd)} =====", flush=True)
        # NO UNC cwd. On the server this repo is \\nasac-m2.unige.ch\... and Windows
        # CreateProcess refuses a UNC working directory outright with WinError 267. The
        # figure scripts resolve everything from __file__ and set their own sys.path, so
        # the working directory is irrelevant to them - any local one will do.
        r = subprocess.run([sys.executable, str(CLUST / cmd[0]), *cmd[1:]],
                           cwd=tempfile.gettempdir(),
                           env={**os.environ, "PYTHONIOENCODING": "utf-8",
                                **({"PAPER2_PAPER": "1"} if a.paper else {})})
        if r.returncode:
            fails.append((i, cmd, r.returncode))
            print(f"  !! step {i} exited {r.returncode}", flush=True)

    print("\n" + "=" * 60)
    if fails:
        print(f"{len(fails)} step(s) FAILED:")
        for i, cmd, rc in fails:
            print(f"  {i:2d}. exit {rc}  {' '.join(cmd)}")
    else:
        print("all steps OK")
    print("figures -> outputs/clustering/paper_figures/   (nothing pushed)")
    if a.paper:
        pdf = paper_preview_pdf()
        if pdf is not None:
            print(f"print-size preview -> {pdf}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
