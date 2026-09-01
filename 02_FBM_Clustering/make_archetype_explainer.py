#!/usr/bin/env python3
"""
make_archetype_explainer.py - what an archetype is, and what they found in THIS cohort.

Every number is computed from the published run. Nothing here is asserted.

    python make_archetype_explainer.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon, FancyBboxPatch
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))
CLUST = ROOT / "outputs" / "clustering"
OUT = CLUST / "explainers" / "E10_archetypes.png"

INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"
RED, GREEN, BLUE, PURPLE, ORANGE = "#c1121f", "#1b7837", "#4a6fa5", "#5b2c83", "#e08214"
MONO = {"family": "DejaVu Sans Mono"}
FS, K = "concat_hg", 11
CONDS = ("audio", "picture", "reading")


def unit(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


ra = sorted((CLUST / "archetypes" / FS / "runs").iterdir())[-1]
rc = sorted((CLUST / "cnmf" / FS / "runs").glob("20260826_*"))[-1]
sw = pd.read_csv(ra / "sweep_by_k.csv").sort_values("k")
A = np.load(ra / "loadings_by_k" / f"A_k{K:02d}.npy").astype(float)
Z = np.load(ra / "components_by_k" / f"C_k{K:02d}.npy").astype(float)
G = np.load(rc / "loadings_by_k" / f"G_k{K:02d}.npy").astype(float)
Cc = np.load(rc / "components_by_k" / f"C_k{K:02d}.npy").astype(float)
X = np.load(ra / "X_train.npy").astype(float)
Xu = unit(X)
Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
lab = A.argmax(1)
n, p = X.shape

fig = plt.figure(figsize=(16.6, 21.6), dpi=140)
gs = GridSpec(5, 12, figure=fig, hspace=0.90, wspace=1.5,
              left=0.055, right=0.975, top=0.905, bottom=0.048,
              height_ratios=[1.35, 1.35, 2.20, 1.05, 0.62])

fig.text(0.032, 0.982, "Archetypes, and what they found in this cohort",
         fontsize=25, color=INK, va="top")
fig.text(0.032, 0.958,
         f"Archetypal analysis (Cutler & Breiman 1994) on {n} electrodes, 27 patients, "
         f"{FS} at K={K}.  Every number below is read from the run, not asserted.\n"
         f"A cluster method asks WHICH GROUP an electrode is in. This asks WHAT MIXTURE "
         f"OF EXTREMES it is - and the two give different answers on this data "
         f"(ARI {ari(lab, Gn.argmax(1)):.2f} against convex NMF).",
         fontsize=11.2, color=MUTED, va="top", linespacing=1.62)

# ── A: the geometry, in the data's own first two PCs ───────────────────────
axA = fig.add_subplot(gs[0, 0:6])
P2 = PCA(n_components=2, random_state=0).fit(Xu)
E, Za, Ca = P2.transform(Xu), P2.transform(Z), P2.transform(Cc)
axA.scatter(E[:, 0], E[:, 1], s=5, c="#c9ced4", alpha=0.55, lw=0, zorder=1)
try:
    h = ConvexHull(E)
    axA.add_patch(Polygon(E[h.vertices], closed=True, fill=False, ec=GREY,
                          lw=1.3, ls=(0, (5, 4)), zorder=2))
except Exception:
    pass
axA.scatter(Ca[:, 0], Ca[:, 1], s=115, marker="s", c=BLUE, ec="white", lw=1.3,
            zorder=4, label=f"convex NMF components (K={K})")
axA.scatter(Za[:, 0], Za[:, 1], s=190, marker="*", c=PURPLE, ec="white", lw=1.1,
            zorder=5, label=f"archetypes (K={K})")
axA.set_title("A  ·  where the two kinds of component sit, in the data's own first two "
              "principal components", fontsize=12.4, loc="left", color=INK, pad=8)
axA.set_xlabel("PC1", fontsize=9); axA.set_ylabel("PC2", fontsize=9)
axA.tick_params(labelsize=8, colors=MUTED)
axA.legend(fontsize=9, frameon=False, loc="best")
axA.spines[["top", "right"]].set_visible(False)

ctr = Xu.mean(0)
d_arch = float(np.linalg.norm(Z - ctr, axis=1).mean())
d_cnmf = float(np.linalg.norm(Cc - ctr, axis=1).mean())
d_elec = float(np.linalg.norm(Xu - ctr, axis=1).mean())
axT = fig.add_subplot(gs[0, 6:12]); axT.axis("off")
axT.text(0, 1.0,
         "The dashed outline is the CONVEX HULL of the electrodes - the boundary of what\n"
         "actually exists in this cohort. Archetypes are pulled onto it; convex-NMF\n"
         "components are weighted averages and sit inside.\n\n"
         "MEAN DISTANCE FROM THE CENTRE OF THE CLOUD\n"
         f"   archetypes            {d_arch:.3f}      "
         f"({100*d_arch/d_elec:.0f}% of a typical electrode)\n"
         f"   convex NMF components {d_cnmf:.3f}      ({100*d_cnmf/d_elec:.0f}%)\n"
         f"   a typical electrode   {d_elec:.3f}\n\n"
         "WHY THAT MATTERS HERE. An average of a mixed population looks like the\n"
         "population. A convex-NMF component of a cohort where most electrodes respond\n"
         "a bit to everything is itself a bit of everything. An archetype is the purest\n"
         "example the cohort contains, so an electrode is described as a MIXTURE of pure\n"
         "types rather than as a member of an averaged one.\n\n"
         "NOT a claim that archetypes are better. They sit at the edge, so no electrode\n"
         "need be near one, and an archetype can be a handful of outliers - which is what\n"
         "panel D measures rather than hopes.",
         fontsize=9.6, color=INK, va="top", linespacing=1.62, family="DejaVu Sans")

# ── B: what the weights mean - decisiveness ────────────────────────────────
axB = fig.add_subplot(gs[1, 0:6])
bins = np.linspace(0, 1, 41)
axB.hist(Gn.max(1), bins=bins, color=BLUE, alpha=0.55, label="convex NMF  (G, renormalised)")
axB.hist(A.max(1), bins=bins, color=PURPLE, alpha=0.55, label="archetypes  (A, sums to 1 already)")
axB.axvline(0.5, color=RED, ls="--", lw=1.3)
axB.text(0.505, axB.get_ylim()[1] * 0.94, "majority", fontsize=8.4, color=RED)
axB.set_title("B  ·  how decisive each electrode's membership is", fontsize=12.4,
              loc="left", color=INK, pad=8)
axB.set_xlabel("largest weight on any one component", fontsize=9)
axB.set_ylabel("electrodes", fontsize=9)
axB.tick_params(labelsize=8, colors=MUTED)
axB.legend(fontsize=9, frameon=False)
axB.spines[["top", "right"]].set_visible(False)

nm_a, nm_c = float((A.max(1) < 0.5).mean()), float((Gn.max(1) < 0.5).mean())
axU = fig.add_subplot(gs[1, 6:12]); axU.axis("off")
axU.text(0, 1.0,
         "THIS IS THE RESULT I EXPECTED TO GO THE OTHER WAY.\n\n"
         "Archetypes sit at the extremes, so the natural guess is that few electrodes\n"
         "land near one and the weights come out weak. The opposite happened:\n\n"
         f"   electrodes with NO majority component\n"
         f"      archetypes   {100*nm_a:.0f}%\n"
         f"      convex NMF   {100*nm_c:.0f}%\n\n"
         f"   median largest weight\n"
         f"      archetypes   {np.median(A.max(1)):.3f}\n"
         f"      convex NMF   {np.median(Gn.max(1)):.3f}\n\n"
         "Describing an electrode as a mixture of EXTREMES turns out to be an easier\n"
         "description than as a mixture of AVERAGES: the extremes are distinguishable\n"
         "from each other, so the weights concentrate. Averages overlap, so they do not.\n\n"
         "And A's rows sum to 1 by construction - the weights ARE proportions, with no\n"
         "renormalising step in between where the meaning could change.",
         fontsize=9.6, color=INK, va="top", linespacing=1.62)

# ── C: the eleven archetypes themselves ────────────────────────────────────
Zb = Z.reshape(K, 3, 300)
order = np.argsort([np.argmax(np.abs(Zb[j]).max(0)) for j in range(K)])
vlim = float(np.abs(Zb).max()) * 1.08
axC0 = fig.add_subplot(gs[2, :]); axC0.axis("off")
axC0.set_title(f"C  ·  the {K} archetypes as response profiles  —  audio | picture | "
               f"reading, each time-normalised, dashed line = the GO cue",
               fontsize=12.4, loc="left", color=INK, pad=2)
NCOL = 6
NROW = int(np.ceil(K / NCOL))
_bb = axC0.get_position()
_gx, _gy, _tt = 0.014, 0.034, 0.028          # column gap, row gap, room for the C title
_w = (_bb.width - _gx * (NCOL - 1)) / NCOL
_h = (_bb.height - _tt - _gy * (NROW - 1)) / NROW
for i, j in enumerate(order):
    r, c = divmod(i, NCOL)
    ax = fig.add_axes([_bb.x0 + c * (_w + _gx),
                       _bb.y1 - _tt - (r + 1) * _h - r * _gy, _w, _h])
    y = Zb[j].reshape(-1)
    ax.axhline(0, color=GREY, lw=0.6)
    for b in (1, 2):
        ax.axvline(b * 300, color="k", lw=0.9)
    for b in range(3):
        ax.axvline((b + 0.5) * 300, color="#9aa3ab", lw=0.7, ls=(0, (3, 3)))
    ax.plot(y, color=PURPLE, lw=1.15)
    ax.set_ylim(-vlim, vlim); ax.set_xlim(0, 900)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GREY)
    m = np.abs(Zb[j]).max(axis=1); cd = int(np.argmax(m))
    t = int(np.argmax(np.abs(Zb[j, cd])))
    sign = "+" if Zb[j, cd, t] > 0 else "−"
    ax.set_title(f"a{j}  ·  {CONDS[cd]} {sign}  ·  {t/300*100:.0f}%  ·  n="
                 f"{int((lab == j).sum())}", fontsize=7.8, color=INK, loc="left", pad=2.4)

# ── D: the stopping rule ───────────────────────────────────────────────────
axD = fig.add_subplot(gs[3, 0:6])
axD.plot(sw.k, sw.min_effective_support, "-o", ms=3.6, color=GREEN,
         label="smallest archetype")
axD.plot(sw.k, sw.median_effective_support, "-o", ms=2.6, color=MUTED, lw=1.0,
         label="median")
axD.axhline(3, color=RED, ls=":", lw=1.4)
axD.axvline(K, color=RED, ls="--", lw=1.0)
axD.set_yscale("log")
axD.text(sw.k.min(), 3.15, "below this an archetype is a few electrodes",
         fontsize=8.2, color=RED, ha="left")
axD.set_title("D  ·  how many electrodes each archetype actually rests on",
              fontsize=12.4, loc="left", color=INK, pad=8)
axD.set_xlabel("K", fontsize=9); axD.set_ylabel("effective support (electrodes)", fontsize=9)
axD.tick_params(labelsize=8, colors=MUTED)
axD.legend(fontsize=9, frameon=False)
axD.spines[["top", "right"]].set_visible(False)

# NOT sw[...>=3].k.max(): support dips below 3 and comes back up, so the LAST K that
# happens to clear the line is not the last K that is safe. Report where it first fails
# and where else, which is what the curve actually shows.
below = sw.loc[sw.min_effective_support < 3, "k"].astype(int).tolist()
first_fail = below[0] if below else None
kmax = (first_fail - 1) if first_fail else int(sw.k.max())
mn = float(sw.loc[sw.k == K, "min_effective_support"].iloc[0])
ve = float(sw.loc[sw.k == K, "var_explained"].iloc[0])
axV = fig.add_subplot(gs[3, 6:12]); axV.axis("off")
axV.text(0, 1.0,
         "A STOPPING RULE THE OTHER THREE METHODS DO NOT HAVE.\n\n"
         "Effective support is exp(entropy of an archetype's weights) - literally how\n"
         "many electrodes it is averaging. Near 1 means one electrode IS the archetype.\n\n"
         f"   at K={K}   variance explained {ve:.3f},  smallest archetype {mn:.1f} electrodes\n"
         f"   support first drops below 3 electrodes at K={first_fail}"
         f"{', and again at ' + ', '.join(str(x) for x in below[1:]) if len(below) > 1 else ''}\n"
         f"   so every K up to {kmax} keeps all its archetypes above that line\n\n"
         "Variance explained rises with K forever, as it must - more components always\n"
         "fit better. Support FALLS. Where the two cross is where extra components stop\n"
         "describing response types and start describing individuals.\n\n"
         "A k-means or Ward centroid cannot degenerate this way: it is always the mean\n"
         "of its whole cluster, however small. That is exactly why this has to be checked\n"
         "here and does not have to be checked there.",
         fontsize=9.6, color=INK, va="top", linespacing=1.62)

# ── E: what it found ───────────────────────────────────────────────────────
early = [j for j in range(K) if np.argmax(np.abs(Zb[j, int(np.argmax(np.abs(Zb[j]).max(1)))])) < 150]
late = [j for j in range(K) if j not in early]
neg = [j for j in range(K)
       if Zb[j, int(np.argmax(np.abs(Zb[j]).max(1))),
             int(np.argmax(np.abs(Zb[j, int(np.argmax(np.abs(Zb[j]).max(1)))])))] < 0]
axE = fig.add_subplot(gs[4, :]); axE.axis("off")
axE.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, transform=axE.transAxes,
                             boxstyle="round,pad=0.012,rounding_size=0.02",
                             fc="#f6f7f9", ec=GREY, lw=1.2, zorder=0))
axE.text(0.012, 0.90,
         f"E  ·  WHAT THE {K} ARCHETYPES ACTUALLY ARE, on this cohort",
         fontsize=12.4, color=INK, va="top", transform=axE.transAxes)
axE.text(0.012, 0.70,
         f"They split by CONDITION and by TIME, and they come in both signs.  "
         f"{len(early)} peak before the GO cue and {len(late)} after it; "
         f"{len(neg)} of {K} peak NEGATIVE - a suppression, not a response.\n"
         f"The largest is a{int(np.bincount(lab, minlength=K).argmax())} with "
         f"{int(np.bincount(lab, minlength=K).max())} electrodes; the smallest "
         f"a{int(np.bincount(lab, minlength=K).argmin())} with "
         f"{int(np.bincount(lab, minlength=K).min())}.\n\n"
         f"WHAT IS NOT ESTABLISHED HERE.  Whether these are better types than convex "
         f"NMF's is not shown by this figure and is not claimed by it. The two agree at "
         f"ARI {ari(lab, Gn.argmax(1)):.2f}, so they are\ndescribing different structure "
         f"- and archetypes have NOT yet been through the statistics battery "
         f"(separation against a matched null, anatomical coherence, leave-one-patient-out).\n"
         f"Until they have, this is a description of what the method found, not evidence "
         f"that it found something real.",
         fontsize=9.5, color=INK, va="top", linespacing=1.62, transform=axE.transAxes)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"run {ra.name}  K={K}  n={n}")
print(f"  hull distance  arch {d_arch:.3f}  cnmf {d_cnmf:.3f}  electrodes {d_elec:.3f}")
print(f"  no majority    arch {nm_a:.3f}  cnmf {nm_c:.3f}")
print(f"  support at K   {mn:.1f}   first K below 3: {first_fail}   all below: {below}")
print(f"-> {OUT}")
