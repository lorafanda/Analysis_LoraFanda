#!/usr/bin/env python3
"""
make_compare_140_figures.py - the 2026-09-18 rerun, seen: overview figures, one figure
per changed contact, and the HG rasters beside them.

Reads what compare_140_trees.py measured (outputs/compare_140/*.tsv) and draws it. Three
kinds of output:

  D1  overview      per patient and per patient x condition: how much changed, how many
                    contacts appeared or disappeared, and whether the two numeric passes
                    (cubes, halves) agree - if they do not, the finding is a bug
  D2  where         in which frequency band and in which half of the warped axis the
                    difference sits, per patient
  case figures      for the most changed contacts: the OLD cube at its full 0-500 Hz, the
                    OLD cube cropped to 0-400 (what the comparison actually uses), the NEW
                    cube, their difference, and underneath the HG trial rasters of both
                    trees - the rasters are drawn from TRIALS, so they change only if the
                    trials or the reference changed, never because of the frequency crop

    python make_compare_140_figures.py                 (overview + 3 cases per patient)
    python make_compare_140_figures.py --cases 6       (more cases per patient)
    python make_compare_140_figures.py --only EL043    (one patient)

Outputs -> outputs/compare_140/figures/
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
CMP = OUTPUTS / "compare_140"
FIGS = CMP / "figures"
NEW_RAW, OLD_RAW = OUTPUTS / "04_ersp_LM_RAWONLY", OUTPUTS / "04_ersp_LM_RAWONLY_old"
NEW_QC, OLD_QC = OUTPUTS / "04_ersp_LM", OUTPUTS / "04_ersp_LM_old"
CONDS = ("audio", "picture", "reading")
DF_HZ = 1000.0 / 256.0
VMAX = 6.0
TOL = 0.01   # dB; below this max|difference| a cube counts as unchanged (see report.md)
INK, MUTED, RED, GREEN, BLUE = "#1b232c", "#68727d", "#c1121f", "#1b7837", "#2471a3"
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.edgecolor": "#c9d1d9", "figure.facecolor": "white"})


def wrap(text, width=150):
    import textwrap
    nl = chr(10)
    return nl.join(nl.join(textwrap.wrap(p, width)) if p.strip() else "" for p in text.split(nl))


def cube_path(tree: Path, pid: str, cond: str, el: str):
    d = tree / pid / "LM" / "ERSP_matrix" / cond
    if not d.is_dir():
        return None
    hits = [f for f in d.glob(f"{pid}_{cond}_*_ERSP_{el}_TN.npy")]
    return hits[0] if hits else None


def hg_path(tree: Path, pid: str, cond: str, el: str):
    d = tree / pid / "LM" / "HG" / cond
    if not d.is_dir():
        return None
    hits = sorted(d.glob(f"{pid}_{cond}_*_HGtrials_{el}.png"))
    return hits[0] if hits else None


def draw_cube(ax, A, title, vmax=VMAX, fmax_label=True):
    if A is None:
        ax.set_axis_off(); ax.text(.5, .5, "not in this tree", ha="center", va="center",
                                   fontsize=8, color=MUTED, transform=ax.transAxes); return None
    im = ax.imshow(A, origin="lower", aspect="auto", cmap="bwr", vmin=-vmax, vmax=vmax,
                   interpolation="none")
    nf, nt = A.shape
    ax.axvline(nt / 2 - .5, color="k", lw=.7, ls=":")
    if fmax_label:
        ticks = [0, 100, 200, 300, 400, 500]
        ticks = [t for t in ticks if t <= (nf - 1) * DF_HZ + 2]
        ax.set_yticks([t / DF_HZ for t in ticks]); ax.set_yticklabels(ticks, fontsize=6.5)
    ax.set_xticks([nt * .25, nt * .75]); ax.set_xticklabels(["stimulus", "post"], fontsize=6.5)
    ax.set_title(title, fontsize=8.5, loc="left")
    return im


def case_figure(row, out: Path):
    pid, cond, el = row.patient, row.condition, row.electrode
    pn, po = cube_path(NEW_RAW, pid, cond, el), cube_path(OLD_RAW, pid, cond, el)
    A = np.load(pn).astype(float) if pn else None
    B = np.load(po).astype(float) if po else None
    nf = min(A.shape[0], B.shape[0]) if (A is not None and B is not None) else None
    hn, ho = hg_path(NEW_QC, pid, cond, el), hg_path(OLD_QC, pid, cond, el)

    fig = plt.figure(figsize=(13.2, 6.6), dpi=150)
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1.15], hspace=0.42, wspace=0.22,
                  left=0.045, right=0.965, top=0.80, bottom=0.05)
    im = draw_cube(fig.add_subplot(gs[0, 0]), B, f"OLD · {B.shape[0] if B is not None else '–'} bins, 0–500 Hz")
    draw_cube(fig.add_subplot(gs[0, 1]), None if B is None else B[:nf], f"OLD cropped to the new axis ({nf} bins)")
    draw_cube(fig.add_subplot(gs[0, 2]), A, f"NEW · {A.shape[0] if A is not None else '–'} bins, 0–400 Hz")
    axd = fig.add_subplot(gs[0, 3])
    if A is not None and B is not None:
        D = A[:nf] - B[:nf]
        v = max(0.1, float(np.nanpercentile(np.abs(D), 99.5)))
        imd = axd.imshow(D, origin="lower", aspect="auto", cmap="PuOr_r", vmin=-v, vmax=v, interpolation="none")
        axd.axvline(D.shape[1] / 2 - .5, color="k", lw=.7, ls=":")
        axd.set_xticks([]); axd.set_yticks([])
        axd.set_title(f"NEW − OLD   ±{v:.2f} dB", fontsize=8.5, loc="left")
        cb = fig.colorbar(imd, ax=axd, fraction=.045, pad=.02); cb.ax.tick_params(labelsize=6)
    else:
        axd.set_axis_off()
    if im is not None:
        cax = fig.add_axes([0.045, 0.455, 0.30, 0.012])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("dB re baseline", fontsize=7)
        cb.ax.tick_params(labelsize=6)

    for k, (p, lab) in enumerate([(ho, "OLD · HG trials"), (hn, "NEW · HG trials")]):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        ax.set_axis_off()
        if p and p.exists():
            ax.imshow(plt.imread(str(p)))
            ax.set_title(f"{lab}  ({p.name})", fontsize=7.5, loc="left", color=MUTED)
        else:
            ax.text(.5, .5, f"{lab}: no figure in this tree", ha="center", va="center",
                    fontsize=9, color=MUTED, transform=ax.transAxes)

    st = (f"{pid} · {cond} · {el}    r {row.r:.4f} · RMSE {row.rmse:.3f} dB · max |Δ| {row.max_abs_diff:.2f} dB · "
          f"HG |Δ| {row.hg_mean_abs_diff:.3f} dB · reference {row.reref_old} → {row.reref_new}\n"
          "The top row compares the cubes: the old one is cropped to the new one's height before the difference, so a "
          "non-zero difference is NOT the 400 Hz crop — it is the bad list, the reference, the triggers or the trials. "
          "The HG rasters below are drawn from trials and are blind to the crop.")
    fig.suptitle(wrap(st), fontsize=9, x=0.045, ha="left", y=0.985)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{pid}_{cond}_{el}.png"
    fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig)
    return f


def overview(C: pd.DataFrame, H: pd.DataFrame, miss: pd.DataFrame):
    ok = C[C.status == "ok"].copy()
    ok["changed"] = ok["max_abs_diff"] > TOL      # not bit-equality: a tolerance, stated
    per = (ok.groupby("patient").agg(n=("key", "size"), changed=("changed", "sum"),
                                     med=("rmse", "median"), mx=("rmse", "max"),
                                     hg=("hg_mean_abs_diff", "median")).reset_index())
    per["pct"] = 100 * per.changed / per.n
    per = per.sort_values("pct", ascending=False)
    pc = ok.pivot_table(index="patient", columns="condition", values="rmse", aggfunc="median")
    pc = pc.reindex(per.patient)

    fig = plt.figure(figsize=(13.5, 7.6), dpi=150)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.35, 1], width_ratios=[1.25, 1, 1],
                  hspace=0.42, wspace=0.3, left=0.075, right=0.975, top=0.84, bottom=0.09)

    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(per))
    ax.barh(y, per.pct, color=[RED if v > 1 else "#cbd5dd" for v in per.pct])
    ax.set_yticks(y); ax.set_yticklabels(per.patient, fontsize=7); ax.invert_yaxis()
    for i, (p_, n_, c_) in enumerate(zip(per.pct, per.n, per.changed)):
        ax.text(min(p_ + 1.5, 100), i, f"{c_}/{n_}", va="center", fontsize=6.5, color=INK)
    ax.set_xlabel("% of contact-conditions whose cube changed at all")
    ax.set_xlim(0, 108)
    ax.set_title("A · who changed", loc="left")

    ax = fig.add_subplot(gs[0, 1])
    M = pc.to_numpy(float)
    im = ax.imshow(np.where(np.isfinite(M), M, np.nan), aspect="auto", cmap="magma_r",
                   vmin=0, vmax=max(0.01, np.nanpercentile(M, 98)))
    ax.set_xticks(range(pc.shape[1])); ax.set_xticklabels(pc.columns, fontsize=7)
    ax.set_yticks(range(len(pc))); ax.set_yticklabels(pc.index, fontsize=6.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]) and M[i, j] > 0:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if M[i, j] > np.nanpercentile(M, 70) else INK)
    fig.colorbar(im, ax=ax, fraction=.045, pad=.02).set_label("median RMSE (dB)", fontsize=7)
    ax.set_title("B · how much, per condition", loc="left")

    ax = fig.add_subplot(gs[0, 2])
    if len(miss):
        g = (miss.groupby(["patient", "present"]).key.nunique().unstack(fill_value=0)
                 .reindex(per.patient).fillna(0))
        for col, c, lab in [("only in the OLD tree", RED, "gone from the new tree"),
                            ("only in the NEW tree", GREEN, "new in the new tree")]:
            if col in g.columns:
                ax.barh(np.arange(len(g)), g[col] if col.endswith("OLD tree") else -g[col],
                        color=c, label=lab)
        ax.axvline(0, color=INK, lw=.8)
        ax.set_yticks(np.arange(len(g))); ax.set_yticklabels(g.index, fontsize=6.5); ax.invert_yaxis()
        ax.set_xlabel("contact-conditions  (left: added · right: dropped)")
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax.set_title("C · what appeared and disappeared", loc="left")

    ax = fig.add_subplot(gs[1, 0])
    if len(H):
        hh = H[H.status == "ok"].groupby(["patient", "condition", "key"]).rmse.mean().rename("halves")
        cc = ok.set_index(["patient", "condition", "key"]).rmse.rename("cubes")
        j = pd.concat([cc, hh], axis=1).dropna()
        ax.scatter(j.cubes, j.halves, s=4, alpha=.3, color=BLUE, edgecolors="none")
        lim = max(j.cubes.max(), j.halves.max()) * 1.05 + 1e-9
        ax.plot([0, lim], [0, lim], color=MUTED, lw=.8, ls="--")
        rho = j.corr(method="spearman").iloc[0, 1]
        ax.set_xscale("symlog", linthresh=1e-3); ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel("pass 1 · cube RMSE (dB)"); ax.set_ylabel("pass 2 · half-cube RMSE (dB)")
        ax.set_title(f"D · the two passes agree (Spearman ρ = {rho:.3f}, n = {len(j)})", loc="left")

    ax = fig.add_subplot(gs[1, 1])
    ch = ok[ok.changed]
    if len(ch):
        bands = [c for c in ok.columns if c.startswith("d_") and c not in ("d_stim", "d_post")]
        vals = [ch[b].mean() for b in bands]
        ax.bar(range(len(bands)), vals, color=BLUE)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels([b[2:].replace("_", "–") + " Hz" for b in bands], fontsize=6.5, rotation=20)
        ax.set_ylabel("mean |Δ| (dB)")
    ax.set_title("E · which band the change is in", loc="left")

    ax = fig.add_subplot(gs[1, 2])
    if len(ch):
        ax.bar([0, 1], [ch.d_stim.mean(), ch.d_post.mean()], color=[BLUE, "#7fb3d5"], width=.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["stimulus half", "post half"], fontsize=7.5)
        ax.set_ylabel("mean |Δ| (dB)")
    ax.set_title("F · which half of the warped axis", loc="left")

    n_id = int((~ok.changed).sum())
    fig.suptitle(wrap(
        "FIG D.1 · what the 2026-09-18 rerun changed — 04_ersp_LM_RAWONLY against 04_ersp_LM_RAWONLY_old, "
        f"{len(ok)} contact-conditions in both trees\n"
        f"{n_id} of them ({100*n_id/len(ok):.1f} %) agree to within {TOL} dB once the old cube is cropped to the new axis. "
        "The rest do not — and the cause is NOT the crop: the adaptive notch is given fmax, so at 500 Hz it removed "
        "harmonics up to 450–483 Hz that it no longer removes, and that filtering changes the time series at every "
        "frequency. See FIG D.2 and attribution.tsv."),
        fontsize=10, x=0.045, ha="left", y=0.985)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "D1_overview.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    return per, pc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=3, help="case figures per patient")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    C = pd.read_csv(CMP / "cubes.tsv", sep="\t")
    H = pd.read_csv(CMP / "halves.tsv", sep="\t") if (CMP / "halves.tsv").exists() else pd.DataFrame()
    miss = pd.read_csv(CMP / "cubes_missing.tsv", sep="\t") if (CMP / "cubes_missing.tsv").exists() else pd.DataFrame()
    print(f"cubes {len(C)} · halves {len(H)} · missing {len(miss)}")
    per, pc = overview(C, H, miss)
    print("\nper patient, most changed first:")
    print(per.round(3).to_string(index=False))

    ok = C[(C.status == "ok") & (C.max_abs_diff > TOL)].copy()
    if a.only:
        ok = ok[ok.patient == a.only]
    picks = (ok.sort_values("rmse", ascending=False)
               .groupby(["patient", "condition"]).head(1)          # the worst per condition
               .sort_values("rmse", ascending=False)
               .groupby("patient").head(a.cases))
    print(f"\n{len(picks)} case figures")
    for _, r in picks.iterrows():
        f = case_figure(r, FIGS / "cases")
        print("  ", f.name, f"rmse {r.rmse:.3f}", flush=True)
    print(f"\nwrote -> {FIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
