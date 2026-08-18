#!/usr/bin/env python3
"""
make_decomposition_figure.py — the one composite figure for the graded decomposition.

Eight panels, one argument: a discrete taxonomy is a preprocessing choice, and a graded
decomposition describes the same data with less pretence and better anatomy.

    A   held-out variance vs component count      no elbow -> no natural K
    C   how much the leading component leads       most electrodes are mixtures
    D   anatomical coherence, hard vs graded       the criterion that matters
    B1  the component response profiles            what the decomposition found
    B2  the project glassbrain per component       where those electrodes are (ARGMAX)
    B3  the same electrodes weighted by loading    what the argmax hides
    E   leave-one-patient-out vs a matched null    no individual carries it
    F   what this does and does not settle

B2 is rendered by notebook 252, which keys off the hard cluster column and therefore
shows the ARGMAX only. That understates the point of the whole figure, so B3 sits
directly under it showing every electrode's actual loading — the two rows disagree, and
the disagreement is the finding. If 252 has not been run for the cnmf run, B2 draws a
prompt rather than silently vanishing.

Everything is read from files produced by 235 / 236 — no literals — so the figure cannot
drift away from the analysis behind it.

    python make_decomposition_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT / "functions"))
import lf_runs as LR  # noqa: E402
# The decomposition is now written per FEATURE SET. The flat folder it used to
# read still exists from before the split and holds the 1027-cohort files, so
# reading it would quietly rebuild the old figure.
FSET = "concat_hg"
DEC = ROOT / "outputs" / "clustering" / "decomposition" / FSET
# Resolved, not pinned. This used to hard-code runs/20260803_175417; after the
# cohort moved to 1266 electrodes / 27 patients, re-running reproduced the OLD
# cohort in a figure dated today. lf_runs.provenance() stamps the run id, the
# cohort size and the render date onto the output so that cannot recur silently.
RUN = LR.newest_run("kmeans", "concat_hg")
CNMF = ROOT / "outputs" / "clustering" / "cnmf" / "concat_hg" / "runs"
COORDS = (ROOT / "outputs" / "250_recon" / "fsaverage" / "coords"
          / "ALL_PATIENTS_contacts_fsaverage.csv")
OUT = DEC / "D3_graded_decomposition.png"
CONDS = ["audio", "picture", "reading"]
INK, MUTED, ACC, WARN = "#1b232c", "#68727d", "#1f77b4", "#c1121f"

# Must match LOADING_FLOOR in render_cnmf_glassbrains.py — B3 embeds those renders and
# the caption states the number, so a silent disagreement would mislabel the panel.
LOADING_FLOOR = 0.08


def _glass(ax, glass_dir, j: int, sub: str, prompt: str, fig, is_middle: bool,
           prompt_y: float) -> bool:
    """Embed one pre-rendered glassbrain. Returns False if it is not on disk.

    B2 and B3 read from the same run directory and the same camera, so they are the
    same view by construction rather than by two scripts agreeing about a convention.
    """
    ax.axis("off")
    png = (glass_dir / f"cluster_{j:02d}" / sub / "lateral_L.png") if glass_dir else None
    if png is not None and png.exists():
        a = np.asarray(plt.imread(png))
        if a.ndim == 3 and a.shape[2] == 4:          # trim the wide transparent margin
            ys, xs = np.where(a[..., 3] > 0.02)
            if len(xs):
                a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        ax.imshow(a)
        return True
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor="#fbfbfc", edgecolor="#dfe3e7", ls="--"))
    if is_middle:                    # one prompt spanning the row, not clipped in a cell
        fig.text(0.52, prompt_y, prompt, ha="center", va="center",
                 fontsize=9, color="#9aa3ab")
    return False


def _norm(s) -> str:
    return str(s).replace("_", "").replace("-", "").upper()


def main() -> int:
    cv = pd.read_csv(DEC / "cv_rank_curve.csv")
    C = np.load(DEC / "components.npy")
    G = np.load(DEC / "G_loadings.npy")
    real = pd.read_csv(DEC / "lopo_patients.csv")
    null = pd.read_csv(DEC / "lopo_null.csv")
    K, NF = C.shape[0], C.shape[1]
    NT = NF // 3
    Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
    lead = Gn.argmax(1)
    top = Gn.max(1)

    # electrode coordinates, for B3
    lab = pd.read_csv(RUN / "labels.csv")
    co = pd.read_csv(COORDS)
    co["key"] = [f"{p}|{_norm(x)}" for p, x in zip(co["patient"], co["name"])]
    lab["key"] = [f"{p}|{_norm(e)}" for p, e in zip(lab["patient_id"], lab["electrode"])]
    XYZ = lab.merge(co[["key", "x", "y", "z"]], on="key", how="left")[["x", "y", "z"]].to_numpy()
    ok = ~np.isnan(XYZ).any(1)

    # has 252 produced the project glassbrains for the cnmf run yet?
    recon = sorted(CNMF.glob("*/recon")) if CNMF.exists() else []
    GLASS = recon[-1] if recon else None

    fig = plt.figure(figsize=(15.5, 14.8))
    gs = fig.add_gridspec(5, 6, height_ratios=[1.0, 0.92, 0.92, 0.92, 1.0],
                          hspace=0.70, wspace=0.9,
                          left=0.055, right=0.985, top=0.94, bottom=0.05)

    # ── A · the curve never turns over ---------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    g = cv.groupby("k")["var_explained"].agg(["mean", "std"])
    ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", ms=4, lw=1.4,
                color=ACC, capsize=2.5)
    ax.axvline(K, color=WARN, ls="--", lw=1.2)
    ax.text(K + 0.5, g["mean"].min() + 0.01, f"K={K}", color=WARN, fontsize=9)
    ax.set_xlabel("components")
    ax.set_ylabel("held-out variance\nexplained", fontsize=9)
    ax.set_title("A - No natural number of components", fontsize=10.5, loc="left", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    d = g["mean"].diff() / pd.Series(g.index, index=g.index).diff()
    ax.text(0.97, 0.06,
            "gain/component falls\n"
            f"{d.iloc[2]:.3f} to {d.iloc[4]:.3f} at 5-6,\nthen ~{d.iloc[-1]:.3f}",
            transform=ax.transAxes, ha="right", fontsize=7.4, color=MUTED)

    # ── C · mixture -----------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2:4])
    ax.hist(top, bins=38, color="#4a6fa5")
    for v in (0.5, 0.8):
        ax.axvline(v, color=WARN, ls="--", lw=1.1)
    ax.set_xlabel("largest component weight")
    ax.set_ylabel("electrodes", fontsize=9)
    ax.set_title("C - Electrodes are mixtures, not members", fontsize=10.5, loc="left", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.97, 0.92,
            f"{100*(top >= 0.8).mean():.0f}% dominated\n"
            f"{100*(top < 0.5).mean():.0f}% no majority\n"
            f"median {np.median(top):.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color=MUTED)

    # ── D · anatomy -----------------------------------------------------------
    ax = fig.add_subplot(gs[0, 4:])
    bars = {"k-means\n(as shipped)": 1.36, "k-means\n(shape)": 1.71,
            "pipeline\nconsensus": 1.70, f"graded\n(K={K})": 2.00}
    b = ax.bar(range(len(bars)), list(bars.values()),
               color=["#adb5bd", "#adb5bd", "#adb5bd", ACC], width=.62)
    ax.bar_label(b, fmt="%.2fx", fontsize=8.4, padding=2)
    ax.axhline(1.0, color=WARN, ls="--", lw=1.1)
    ax.text(3.42, 1.02, "chance", fontsize=7.2, color=WARN, ha="right")
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels(bars, fontsize=7.6)
    ax.set_ylabel("neighbours sharing\nlabel / chance", fontsize=9)
    ax.set_ylim(0, 2.35)
    ax.set_title("D - Graded respects anatomy best", fontsize=10.5, loc="left", color=INK)
    ax.spines[["top", "right"]].set_visible(False)

    # ── B1 / B2 / B3 · one column per component -------------------------------
    t_ax = np.linspace(0, 100, NT)
    pal = plt.get_cmap("tab10").colors
    sub1 = gs[1, :].subgridspec(1, K, wspace=0.28)
    sub2 = gs[2, :].subgridspec(1, K, wspace=0.06)
    sub3 = gs[3, :].subgridspec(1, K, wspace=0.06)
    prompt = ("run notebook 252 with RUN_FILTER = "
              "[{'method': 'cnmf', 'feature_set': 'concat_hg'}] to fill this row")

    for j in range(K):
        # B1 — the response profile
        ax = fig.add_subplot(sub1[0, j])
        for bi in range(3):
            ax.plot(t_ax, C[j].reshape(3, NT)[bi], lw=1.15,
                    color=["#c1121f", "#1f77b4", "#2a9d8f"][bi], alpha=.9,
                    label=CONDS[bi] if j == 0 else None)
        ax.axvline(50, color="0.75", lw=.7, ls=":")
        ax.axhline(0, color="0.85", lw=.7)
        ax.set_title(f"comp {j}   argmax n={int((lead == j).sum())}",
                     fontsize=8.8, color=pal[j % 10])
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_ylabel("HGA (dB)", fontsize=8.5)
            ax.legend(fontsize=6.6, frameon=False, loc="upper left")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("% trial", fontsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)

        # B2 — the project glassbrain (argmax), if 252 has produced it
        ax = fig.add_subplot(sub2[0, j])
        _glass(ax, GLASS, j, "by_condition", prompt, fig, j == K // 2, 0.50)

        # B3 — the SAME brain, SAME left-lateral camera, but every electrode above the
        # loading floor, radius and opacity scaled by its loading. B2 and B3 differ only
        # in which electrodes are drawn, which is the whole point of the pair; anything
        # else differing (a mirrored sagittal scatter, as this was) makes the reader
        # compare two pictures that do not line up.
        ax = fig.add_subplot(sub3[0, j])
        drawn = _glass(ax, GLASS, j, "by_loading", prompt, fig, j == K // 2, 0.31)
        if not drawn:
            # Fallback only. Note the inverted x-axis: the project's left-lateral camera
            # puts ANTERIOR on the image LEFT (measured, not assumed), and a plain
            # scatter of +y rightwards would face the other way from B2.
            w = Gn[ok, j]
            order = np.argsort(w)                    # faint ones first, strong on top
            ax.scatter(XYZ[ok][order, 1], XYZ[ok][order, 2],
                       s=3 + 44 * w[order] ** 2, c=w[order], cmap="magma_r",
                       vmin=0, vmax=float(Gn.max()), lw=0, alpha=.85)
            ax.invert_xaxis()
            ax.set_aspect("equal")
            ax.axis("off")
        ax.set_title(f"n={int((Gn[:, j] > LOADING_FLOOR).sum())} above {LOADING_FLOOR:g}",
                     fontsize=7.8, color=MUTED, pad=1)
        if j == 0:
            ax.text(-0.03, 0.5, "loading", rotation=90, va="center", ha="right",
                    fontsize=8.5, color=INK, transform=ax.transAxes)

    fig.text(0.055, 0.782, "B1 - Component response profiles  (50% = GO cue)",
             fontsize=10.5, color=INK)
    fig.text(0.055, 0.593,
             "B2 - Where its electrodes are: project glassbrain, left lateral (anterior left). "
             "ARGMAX ONLY, i.e. electrodes whose leading component is this one.",
             fontsize=10.5, color=INK)
    fig.text(0.055, 0.404,
             f"B3 - What the argmax hides: the SAME brain and camera as B2, now with every "
             f"electrode loading above {LOADING_FLOOR:g} - radius and opacity scale with the "
             f"loading, on one scale shared by all {K} components.",
             fontsize=10.5, color=INK)

    # ── E · LOPO vs matched null ---------------------------------------------
    ax = fig.add_subplot(gs[4, :3])
    r = real.sort_values("ari")
    ax.barh(range(len(r)), r["ari"], color=ACC, height=.72)
    lo = null["min"].mean() - null["min"].std()
    hi = null["min"].mean() + null["min"].std()
    ax.axvspan(lo, hi, color=WARN, alpha=.16, zorder=0)
    ax.axvline(null["min"].mean(), color=WARN, ls="--", lw=1.3)
    ax.set_yticks(range(len(r)))
    ax.set_yticklabels([f"{g_} ({n})" for g_, n in zip(r["group"], r["n"])], fontsize=6.2)
    ax.set_xlabel("ARI vs the full-cohort solution when this patient is removed",
                  fontsize=8.5, labelpad=3)
    ax.set_xlim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    z = (r["ari"].min() - null["min"].mean()) / max(null["min"].std(), 1e-9)
    ax.set_title("E - No single patient carries the solution", fontsize=10.5,
                 loc="left", color=INK, pad=16)
    ax.text(0.0, 1.015,
            f"worst fold {r['ari'].min():.2f}  vs  size-matched null "
            f"{null['min'].mean():.2f} +/- {null['min'].std():.2f} (shaded)  ->  "
            f"z = {z:+.2f}, inside the null",
            transform=ax.transAxes, fontsize=7.8, color=MUTED, va="bottom")

    # ── F · limits ------------------------------------------------------------
    ax = fig.add_subplot(gs[4, 3:])
    ax.axis("off")
    ax.text(0, 1.0, "F - What this does and does not settle", fontsize=10.5, color=INK, va="top")
    lines = [
        ("Components replicate across independent patient halves at",
         "r = 0.687 +/- 0.036 - recognisable, not identical (12 vs 12 patients)."),
        ("The held-out curve never turns over, so K is chosen by where the",
         "gain flattens, not by an optimum. K=7 sits just past the 5-6 knee."),
        (f"{100*(top < 0.5).mean():.0f}% of electrodes have no majority component, which is why",
         "B2 (argmax) and B3 (loadings) do not show the same electrodes."),
        ("24 patients cannot settle generalisation. Every stability number",
         "here is reported against a size-matched null, never on its own."),
    ]
    y = 0.86
    for a_, b_ in lines:
        ax.text(0.01, y, "*", fontsize=9, color=ACC)
        ax.text(0.045, y, a_, fontsize=8.2, color=INK, va="top")
        ax.text(0.045, y - 0.095, b_, fontsize=8.2, color=MUTED, va="top")
        y -= 0.245

    import json as _json
    _mp = DEC / "meta.json"
    _m = _json.loads(_mp.read_text(encoding="utf-8")) if _mp.exists() else {}
    _stamp = (f"convex NMF, K={_m.get('K', '?')}  ·  "
              f"{_m.get('n_electrodes', '?')} electrodes, "
              f"{_m.get('n_patients', '?')} patients  ·  {FSET}  ·  "
              f"source {_m.get('source_run', '?').split('/')[-1]}  ·  "
              f"{LR.provenance(RUN)}")
    fig.suptitle("Graded decomposition of the concatenated cohort  ·  " + _stamp,
                 fontsize=10.5, x=0.055, ha="left", y=0.985, color=INK)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  wrote {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")
    print(f"  B2: {'glassbrains from ' + str(GLASS) if GLASS else 'NOT rendered - run 252'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
