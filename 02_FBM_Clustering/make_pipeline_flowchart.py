#!/usr/bin/env python3
"""
make_pipeline_flowchart.py - every file you have to run to rebuild the pipeline, in order.

Read off the scripts rather than remembered: the edges below are the ones the code
actually has - which file reads which artifact - not a tidied-up version of them. Where
an ordering constraint exists it is drawn as an arrow and named, because the two that
bit hardest this month were both orderings that looked like independence.

    python make_pipeline_flowchart.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "clustering" / "explainers" / "E9_pipeline_flowchart.png"

INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"
RED, GREEN, BLUE, PURPLE, ORANGE = "#c1121f", "#1b7837", "#4a6fa5", "#5b2c83", "#e08214"
STAGE = {"01": "#2c7fb8", "cohort": "#8a6d3b", "fit": "#41ab5d",
         "stats": "#5b2c83", "assets": "#e08214", "recon": "#0b7a75", "web": "#c0392b"}
MONO = {"family": "DejaVu Sans Mono"}

W, H = 20.4, 28.2
fig = plt.figure(figsize=(W, H), dpi=132)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box_h(sub, sfs=8.1):
    """The height this caption needs.

    Hand-set heights were the bug: every box with three or four caption lines had the
    last one or two printed below its own border. One data unit is one inch here, so a
    line is sfs * linespacing / 72 inches, plus the title band and a little padding.
    """
    n = sub.count("\n") + 1 if sub else 0
    return 0.52 if not n else 0.72 + n * sfs * 1.42 / 72 + 0.16


def box(x, ytop, w, title, sub="", *, color=BLUE, fill="#ffffff", lw=1.8,
        tfs=10.6, sfs=8.1, mono_title=True, dashed=False, h=None):
    h = box_h(sub, sfs) if h is None else h
    y = ytop - h
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.055,rounding_size=0.1",
                                fc=fill, ec=color, lw=lw,
                                ls=(0, (5, 3)) if dashed else "solid", zorder=3))
    ty = y + h - 0.32 if sub else y + h / 2
    ax.text(x + 0.18, ty, title, fontsize=tfs, color=INK, va="center", zorder=4,
            **(MONO if mono_title else {}))
    if sub:
        ax.text(x + 0.18, y + h - 0.68, sub, fontsize=sfs, color=MUTED, va="top",
                zorder=4, linespacing=1.42)
    return (x, y, w, h)


def arrow(a, b, label="", *, color=GREY, side="v", lw=1.7, dx=0.0, dy=0.0,
          lfs=7.6, lcol=None, style="-|>", dashed=False):
    """a, b are (x, y, w, h). side 'v' = bottom of a to top of b, 'h' = right to left."""
    if side == "v":
        p0 = (a[0] + a[2] / 2 + dx, a[1])
        p1 = (b[0] + b[2] / 2 + dx, b[1] + b[3])
    else:
        p0 = (a[0] + a[2], a[1] + a[3] / 2 + dy)
        p1 = (b[0], b[1] + b[3] / 2 + dy)
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=15,
                                 color=color, lw=lw, zorder=2,
                                 linestyle=(0, (4, 3)) if dashed else "solid",
                                 shrinkA=2, shrinkB=2))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx + (0.14 if side == "v" else 0), my + (0 if side == "v" else 0.16),
                label, fontsize=lfs, color=lcol or MUTED, va="center",
                ha="left" if side == "v" else "center", zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.2))


def band(y, h, name, color):
    ax.add_patch(FancyBboxPatch((0.30, y), W - 0.60, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="none", alpha=0.055, zorder=0))
    ax.text(0.52, y + h - 0.26, name, fontsize=11.4, color=color, va="top",
            fontweight="bold", zorder=1)


fig.text(0.026, 0.988, "Rebuilding the pipeline, end to end",
         fontsize=25, color=INK, va="top")
fig.text(0.026, 0.9695,
         "Every file that has to be run, in the order the data forces. Solid arrows are "
         "hard dependencies - the target reads what the source wrote. Dashed arrows are "
         "orderings that are easy to miss\nbecause the steps look independent. Boxes are "
         "the thing you actually run; the grey text under each arrow is the artifact it "
         "passes on.",
         fontsize=10.4, color=MUTED, va="top", linespacing=1.62)

# ── 01 preprocessing ────────────────────────────────────────────────────────
band(23.55, 3.30, "01_FBM_Analysis   ·   signal to ERSP", STAGE["01"])
a1 = box(0.9, 26.12, 5.4, "raw sEEG  +  prep0 TSVs",
         color=GREY, fill="#f6f7f9", mono_title=False, tfs=10.2)
a2 = box(0.9, 25.44, 5.4, "adjust_fixation_cross_duration.py",
         "trial_end = next onset - (1.8 - U(0,0.2))s\nONE-TIME per patient. Has --restore "
         "and a .bak guard;\nrunning it twice is refused, not silently doubled.",
         color=STAGE["01"], tfs=9.6, sfs=7.6)
a3 = box(7.2, 25.44, 6.2, "150_ERSP_analysis_pipeline_noTwarping.ipynb",
         "the ERSP itself: resample 1000 Hz, STFT nperseg=128 / nfft=256 /\n"
         "noverlap=108, baseline (-0.6,-0.1), time-normalise to 300 bins",
         color=STAGE["01"], tfs=9.6, sfs=7.6)
a4 = box(14.3, 25.44, 5.2, "move_halves_out_of_ersp_matrix.py",
         "_half1/_half2 OUT of ERSP_matrix, into ERSP_halves.\n"
         "SKIP THIS AND THE COHORT TRIPLES - the loader globs\n"
         "ERSP_matrix and counts each electrode three times.",
         color=RED, tfs=9.6, sfs=7.6)
a5 = box(7.2, 24.06, 6.2, "outputs/04_ersp_LM_RAWONLY/<pt>/LM/ERSP_matrix/",
         color=GREY, fill="#f6f7f9", tfs=8.8)
arrow(a1, a2, "", side="v")
arrow(a2, a3, "corrected trial windows", side="h", dy=0)
arrow(a3, a4, "9,342 cubes", side="h")
arrow(a4, a5, "", side="v", color=RED, dashed=True)

# ── the cohort ──────────────────────────────────────────────────────────────
band(21.55, 1.42, "the cohort   ·   built once, serially", STAGE["cohort"])
b1 = box(6.2, 22.77, 8.0, "rebuild_concat_cache.py --apply",
         "SERIAL AND FIRST. Writes a NEW cache dir - prepare_dataset decides a cache hit "
         "on params\nalone, which carry no patient list, so a changed cohort in the same "
         "dir is read as a hit.\n-> outputs/_dataset/concat_source_v4/   (1693 gated / "
         "2959 ungated, 27 patients)",
         color=STAGE["cohort"], tfs=10.4, sfs=7.7)
arrow(a5, b1, "", side="v", dx=1.0)

# ── the fits ────────────────────────────────────────────────────────────────
band(16.30, 5.00, "the fits   ·   240 / 241 in parallel, then 242, then 243", STAGE["fit"])
c1 = box(0.9, 20.55, 4.3, "240_cluster_kmeans.ipynb",
         "k-means, K=5..30, every\nfeature set. SKIP_IF_RUN_EXISTS\nleaves a fitted set "
         "alone.", color=STAGE["fit"], tfs=10.0, sfs=7.7)
c2 = box(5.6, 20.55, 4.3, "241_cluster_hierarchical.ipynb",
         "Ward. Independent of 240 -\nrun them at the same time.",
         color=STAGE["fit"], tfs=9.4, sfs=7.7)
c3 = box(10.3, 20.55, 4.3, "242_cluster_cnmf.ipynb",
         "run_decomposition ->\npublish_decomposition ->\nsweep_decomposition",
         color=STAGE["fit"], tfs=10.0, sfs=7.7)
c4 = box(15.0, 20.55, 4.5, "243_cluster_archetypes.ipynb",
         "run_archetypes.py.\nOptional fourth track -\nit does not gate 249.",
         color=STAGE["fit"], tfs=9.8, sfs=7.7)
arrow(b1, c1, "", side="v", dx=-2.4)
arrow(b1, c2, "", side="v", dx=-0.5)
arrow(c1, c3, "", side="h", dashed=True, color=RED, lw=2.0)
ax.text(10.15, 20.92, "cNMF and archetypes take X_train FROM THE K-MEANS RUN.\n"
                      "For a NEW feature set 242 and 243 cannot start until 240 has\n"
                      "finished it - the three are only parallel for sets 240 already has.",
        fontsize=8.0, color=RED, ha="right", va="bottom", linespacing=1.45, zorder=6,
        bbox=dict(fc="white", ec=RED, lw=1.0, alpha=0.95, pad=3.0))
arrow(c3, c4, "", side="h", dashed=True, color=RED, lw=2.0)

c5 = box(2.6, 18.87, 6.4, "cell 7 of 240 / 241 / 242",
         "make_heldout_variance.py --from-cache ...  ->  heldout_variance_<method>.csv\n"
         "bi-cross-validated, the only curve that can turn over and so choose K",
         color=STAGE["fit"], tfs=9.8, sfs=7.6)
c6 = box(10.3, 18.87, 9.2, "sweep_stability.py --new-concat   [--native]",
         "stability_by_k.csv, and with --native the refit uses the run's OWN method\n"
         "-> stability_by_k_native.csv. On cNMF that roughly doubles the number.",
         color=STAGE["fit"], tfs=9.8, sfs=7.6)
arrow(c1, c5, "", side="v", dx=0.6)
arrow(c3, c6, "", side="v", dx=1.0)

c7 = box(2.6, 17.17, 16.9,
         "outputs/clustering/<method>/<feature_set>/runs/<YYYYmmdd_HHMMSS>/",
         color=GREY, fill="#f6f7f9", tfs=9.4)
arrow(c5, c7, "", side="v", dx=1.0)

# ── statistics ──────────────────────────────────────────────────────────────
band(12.05, 4.20, "249   ·   statistics and the comparison figures", STAGE["stats"])
d1 = box(0.9, 15.60, 5.6, "249  §1  merge the sweeps",
         "heldout_variance_ALL.csv, then the K where\nconvex NMF's held-out curve PEAKS, "
         "per feature set\n-> peak_k.json   (concat_hg 11, concat_rawds 12)",
         color=STAGE["stats"], tfs=10.0, sfs=7.6, mono_title=False)
d2 = box(7.0, 15.60, 6.1, "249  §2  make_cluster_statistics.py",
         "separation vs a matched null, anatomical coherence,\nleave-one-patient-out, "
         "agreement. 50 nulls per cell.\n-> statistics/<fs>_K<k>/  +  cluster_statistics.json",
         color=STAGE["stats"], tfs=10.0, sfs=7.6, mono_title=False)
d3 = box(13.6, 15.60, 5.9, "249  §3  the figures",
         "make_cluster_figures.py -> C.3a/b/c, C.8a/b/c\nmake_heldout_figure.py  -> C.13\n"
         "cut at peak_k, in the visualizer's own palette",
         color=STAGE["stats"], tfs=10.0, sfs=7.6, mono_title=False)
arrow(c7, d1, "", side="v", dx=-6.0)
arrow(d1, d2, "peak K", side="h")
arrow(d2, d3, "the stats", side="h")

d4 = box(2.6, 14.32, 16.9, "VALID AT ONE K ONLY",
         "Every number in §2 is scored against a null REFITTED AT THAT K, so it means "
         "nothing at another cut. cluster_statistics.json\ncarries the K it was computed "
         "at and the report checks it before showing a row - which is why the statistics "
         "section\ngoes quiet when you move the K control, and now says which K they "
         "belong to instead of just vanishing.\n\n"
         "Re-running a feature set does NOT update its run, it SUPERSEDES it: run ids are "
         "timestamps and every resolver takes\nthe newest. The old run keeps its "
         "statistics, its stability sweeps and its place in the coverage manifest, and "
         "nothing\nresolves to it any more. SKIP_IF_RUN_EXISTS in the notebooks and "
         "--force on run_archetypes exist for exactly this.",
         color=RED, fill="#fdf3f3", tfs=10.4, sfs=8.0, mono_title=False)
arrow(d2, d4, "", side="v", dx=1.0, color=RED)

# ── per-run assets ──────────────────────────────────────────────────────────
band(9.30, 2.60, "per-run assets   ·   what the report embeds", STAGE["assets"])
e1 = box(0.9, 11.27, 6.0, "make_missing_centroids.py",
         "one centroid chip per cluster PER K (5..30).\nconcat_hg: mean dB with +/-1 SD.\n"
         "concat_rawds / bands5: averaged ERSP\nwith a per-bin SD dot.",
         color=STAGE["assets"], tfs=10.2, sfs=7.7)
e2 = box(7.4, 11.27, 6.0, "make_centroid_rasters.py",
         "the second view of each cluster.\nconcat_hg: EVERY electrode as a row,\n"
         "sorted by membership. rawds: the same\nplane with a proper SD legend.",
         color=STAGE["assets"], tfs=10.2, sfs=7.7)
e3 = box(13.9, 11.27, 5.6, "252_clustering_recon.ipynb",
         "per-cluster glassbrains,\nby condition and by patient.\n"
         "Needs 251's fsaverage coords.",
         color=STAGE["assets"], tfs=10.0, sfs=7.7)
for src, dst in ((c7, e1), (c7, e2), (c7, e3)):
    arrow(src, dst, "", side="v", dx=(dst[0] + dst[2] / 2) - (c7[0] + c7[2] / 2))

# ── recon + bundle ──────────────────────────────────────────────────────────
band(5.55, 3.30, "recon and the bundle   ·   what makes the visualizer work", STAGE["recon"])
f0 = box(0.9, 8.12, 5.6, "251_recon_shared_data.ipynb",
         "ONE-TIME per patient. fsaverage meshes +\nALL_PATIENTS_contacts_fsaverage.csv",
         color=STAGE["recon"], tfs=9.8, sfs=7.6)
f1 = box(7.4, 8.12, 12.1, "make_coverage_bundle.py",
         "THE VISUALIZER'S RUN LIST LIVES HERE. It builds coverage_viz/manifest.json and "
         "the per-run arrays;\na run that is not in this file cannot be opened at all, "
         "however complete it is on disk.\n"
         "Re-run it after EVERY new clustering run, or the page silently offers the old "
         "cohort.",
         color=STAGE["recon"], tfs=10.6, sfs=7.8)
arrow(f0, f1, "meshes + coords", side="h")
arrow(e3, f1, "", side="v", dx=2.8)
arrow(c7, f1, "", side="v", dx=6.6, dashed=False)

f2 = box(4.6, 6.63, 11.2, "clustering_visualizer.html  +  its exported report",
         "reads the bundle, the run directories and the centroid chips - all over HTTP, "
         "from the repository",
         color=STAGE["recon"], fill="#eef7f6", tfs=11.0, sfs=8.0)
arrow(f1, f2, "", side="v", dx=-1.6)

# ── the website ─────────────────────────────────────────────────────────────
band(2.40, 2.95, "analysis_status.html   ·   the written record", STAGE["web"])
g1 = box(0.9, 4.77, 5.8, "make_cluster_webblock.py --insert",
         "the 02 tab's statistics block, generated\nfrom the files - prose included",
         color=STAGE["web"], tfs=9.6, sfs=7.6)
g2 = box(7.2, 4.77, 5.8, "make_s2_gallery.py --insert",
         "one example of every figure the stage\nmakes; checks git, not the disk",
         color=STAGE["web"], tfs=9.6, sfs=7.6)
g3 = box(13.5, 4.77, 6.0, "make_kiss_tab.py --insert",
         "the plain-words tab; also\nmake_bands_figure / the explainers",
         color=STAGE["web"], tfs=9.6, sfs=7.6)
for s_ in (g1, g2, g3):
    arrow(d3, s_, "", side="v", dx=(s_[0] + s_[2] / 2) - (d3[0] + d3[2] / 2))

g4 = box(2.6, 3.65, 16.9, "git add  ·  git commit  ·  git push       <-- NOT OPTIONAL",
         "THE SITE AND THE VISUALIZER FETCH EVERYTHING FROM THE REPOSITORY. A figure, a "
         "CSV or a manifest that exists\nonly on your disk is a 404 to both of them - "
         "which is what \"No metrics published for K=9\" and the gallery's\nbroken boxes "
         "both were. Committing is a step in the pipeline, not tidying up afterwards.",
         color=RED, fill="#fdf3f3", tfs=11.4, sfs=8.2, mono_title=False)
arrow(g2, g4, "", side="v", dx=1.0, color=RED)
arrow(f2, g4, "", side="v", dx=-4.0, color=RED, dashed=True)

# ── the short version ───────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0.60, 0.30), W - 1.20, 1.92,
                            boxstyle="round,pad=0.05,rounding_size=0.1",
                            fc="#f6f7f9", ec=GREY, lw=1.4, zorder=3))
ax.text(0.95, 2.02, "THE SHORT VERSION, if nothing has changed but the cohort",
        fontsize=11.6, color=INK, va="top", zorder=4)
steps = [
    "1.  adjust_fixation_cross_duration.py   ->   150_ERSP...ipynb   ->   "
    "move_halves_out_of_ersp_matrix.py",
    "2.  python rebuild_concat_cache.py --apply                      "
    "(serial, alone, first)",
    "3.  240 + 241 together;  then 242;  then 243                    "
    "(242/243 read 240's run - not parallel for a NEW feature set)",
    "4.  cell 7 of 240 / 241 / 242      then   python sweep_stability.py "
    "--new-concat --native",
    "5.  249                                                          "
    "(merge -> peak K -> statistics -> figures)",
    "6.  make_missing_centroids.py  ·  make_centroid_rasters.py  ·  252  ->  "
    "make_coverage_bundle.py",
    "7.  make_cluster_webblock / make_s2_gallery / make_kiss_tab  --insert   "
    "->   commit and push BOTH repos",
]
for i, t in enumerate(steps):
    ax.text(0.95, 1.70 - i * 0.205, t, fontsize=8.6, color=INK, va="top",
            zorder=4, **MONO)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=132, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"-> {OUT}")
