#!/usr/bin/env python3
"""
make_dpca_explainer.py - how demixed PCA works, on a feature set small enough to read.

The same twelve invented electrodes as E1, now recorded in THREE conditions. Two shapes are
planted: a late burst that is the same in every condition (what an electrode does whatever
the input) and an early transient present in the audio condition only (what depends on
the input). Each electrode is a known mixture of the two plus noise, so the recovered
components and weights can be checked against what it was built from.

WHAT IT IS MEANT TO MAKE OBVIOUS

  * dPCA does ONE THING before any fitting: it splits the data into two parts with the
    condition labels. The average over the three conditions is the condition-independent
    part; what each condition keeps after that average is subtracted is the
    condition-dependent part. Nothing is estimated here - it is arithmetic on the labels.
  * then it finds a few components INSIDE each part. A component has a time course (one
    per condition in the condition part) and a signed weight on every electrode.
  * a weight is signed and a component is a direction, not a recordable response: an
    electrode's record is weight x time course, summed over components.
  * the honest cost of the split: a response that exists in one condition only is cut
    in three - one third of it counts as common, two thirds as specific. Panel B shows
    that happen to the audio transient.

Fitted with the project's own run_counter_dpca.dpca_fit, not a re-implementation.

    python make_dpca_explainer.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))
import lf_decompose as LD                                                        # noqa: E402
_spec = importlib.util.spec_from_file_location("run_counter_dpca", ROOT / "run_counter_dpca.py")
RD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RD)

OUT = ROOT / "outputs" / "clustering" / "explainers"
N_ELEC, N_TIME = 12, 40
CONDS = ["audio", "picture", "reading"]
CC = {"audio": "#e06c9f", "picture": "#4a6fa5", "reading": "#8c6d46"}
INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"
COL_T, COL_C = "#2a9d5c", "#5b2c83"


def build():
    t = np.linspace(0, 1, N_TIME)
    burst = np.exp(-((t - 0.70) ** 2) / 0.012)                 # same in every condition
    trans = np.exp(-((t - 0.20) ** 2) / 0.006)                 # audio only
    w_t = np.array([1.0, 0.9, 0.8, 0.6, 0.3, 0.5, 0.0, 0.0, 0.0, 0.2, 0.0, 0.4])
    w_c = np.array([0.0, 0.0, 0.2, 0.4, 0.9, 0.5, 1.0, 0.8, 0.7, 0.9, 0.2, 0.3])
    amp = np.linspace(1.0, 2.2, N_ELEC)
    rng = np.random.default_rng(3)
    X = np.zeros((N_ELEC, 3, N_TIME))
    for c in range(3):
        X[:, c, :] = (w_t[:, None] * burst[None, :] + (w_c[:, None] * trans[None, :] if c == 0 else 0.0)) * amp[:, None]
    clean = X.copy()
    X = X + rng.normal(0, 0.05, X.shape)
    return t, burst, trans, w_t, w_c, clean, X


def main() -> int:
    t, burst, trans, w_t, w_c, clean, X = build()
    C, T = 3, N_TIME
    Xu = LD.unit_norm(X.reshape(N_ELEC, -1))                   # what the analysis fits on: shape only
    Xc = Xu - Xu.mean(1, keepdims=True)                        # centred per electrode
    fit = RD.dpca_fit(Xc, C, T, {"time": 1, "condition": 1}, 1e-3)
    F, Dm, Z, marg = fit["F"], fit["D"], fit["Z"], fit["marg"]
    i_t, i_c = marg.index("time"), marg.index("condition")
    # sign convention for the picture only: make the time component's burst positive and the
    # condition component's audio transient positive (a component and its weights can flip together)
    for i, ref in ((i_t, burst), (i_c, trans)):
        if np.dot(Z[i, 0], ref) < 0:
            Z[i] *= -1; F[:, i] *= -1; Dm[i] *= -1
    # the planted truth, marginalized the same way (without noise), for the grey lines
    cu = LD.unit_norm(clean.reshape(N_ELEC, -1)).reshape(N_ELEC, C, T)
    cu = cu - cu.mean((1, 2), keepdims=True)
    truth_t = cu.mean(1)                                       # (N, T) the common part per electrode
    truth_c = cu - truth_t[:, None, :]                         # (N, C, T) the condition part
    margs = RD.marginalize(Xc, C, T)
    tot = float((Xc ** 2).sum())
    var_by = {k: float((v ** 2).sum()) / tot for k, v in margs.items()}
    pick = 5
    recon = (F @ (Dm @ Xc)).reshape(N_ELEC, C, T)
    err = float(np.abs(Xc.reshape(N_ELEC, C, T) - recon).mean())

    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.2, 9.2), dpi=200)
    gs = GridSpec(2, 3, hspace=0.55, wspace=0.32, left=0.055, right=0.975, top=0.600, bottom=0.070)
    Xr = Xu.reshape(N_ELEC, C, T)

    # A - the data: 12 electrodes, three conditions each
    ax = fig.add_subplot(gs[0, 0])
    for i in range(N_ELEC):
        for c, cond in enumerate(CONDS):
            ax.plot(t, Xr[i, c] + i * 0.30, color=CC[cond], lw=0.9, alpha=0.9)
        ax.text(-0.04, i * 0.30 + 0.05, f"e{i}", fontsize=6.6, color=MUTED, ha="right", va="center")
    for c, cond in enumerate(CONDS):
        ax.text(0.02 + 0.3 * c, N_ELEC * 0.30 + 0.30, cond, color=CC[cond], fontsize=7.5)
    ax.set_title("A · the data — 12 electrodes × 3 conditions × 40 points\nunit-normed, so only shape is left",
                 fontsize=9.5, loc="left", color=INK, pad=5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.08, 1.02)
    for s in ax.spines.values():
        s.set_visible(False)

    # B - the split, on e5: average over conditions, and what each condition keeps
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1], hspace=0.45)
    axb1 = fig.add_subplot(sub[0]); axb2 = fig.add_subplot(sub[1])
    xc5 = Xc.reshape(N_ELEC, C, T)[pick]
    for c, cond in enumerate(CONDS):
        axb1.plot(t, xc5[c], color=CC[cond], lw=1.0)
    axb1.plot(t, xc5.mean(0), color=INK, lw=2.2, label="mean of the three = the common part")
    axb1.legend(fontsize=6.4, frameon=False, loc="upper left")
    for c, cond in enumerate(CONDS):
        axb2.plot(t, xc5[c] - xc5.mean(0), color=CC[cond], lw=1.4)
    axb2.axhline(0, color=GREY, lw=0.8)
    axb2.text(0.02, 0.86, "each minus the mean = the condition part", transform=axb2.transAxes, fontsize=6.4, color=INK)
    axb1.set_title("B · the split, on e5 — no fitting yet, only the labels\n"
                   "top: the common part · bottom: the condition part", fontsize=9.5, loc="left", color=INK, pad=5)
    for a in (axb1, axb2):
        a.set_xticks([]); a.set_yticks([])
        a.spines[["top", "right"]].set_visible(False)

    # C - the recovered components against the planted parts
    axc = fig.add_subplot(gs[0, 2])
    gt = truth_t[pick] / np.linalg.norm(truth_t[pick]); zt = Z[i_t, 0] / np.linalg.norm(Z[i_t, 0])
    axc.plot(t, gt + 0.55, color=GREY, lw=2.6); axc.plot(t, zt + 0.55, color=COL_T, lw=1.5)
    axc.text(1.01, 0.62, "common\ncomponent", color=COL_T, fontsize=7.5, va="center")
    for c, cond in enumerate(CONDS):
        gc = truth_c[pick, c] / np.linalg.norm(truth_c[pick]); zc = Z[i_c, c] / np.linalg.norm(Z[i_c])
        axc.plot(t, gc, color=GREY, lw=2.6); axc.plot(t, zc, color=CC[cond], lw=1.5)
    axc.text(1.01, 0.05, "condition\ncomponent", color=COL_C, fontsize=7.5, va="center")
    axc.set_title("C · the recovered components\ngrey = the planted parts (e5's, scaled)", fontsize=9.5, loc="left", color=INK, pad=5)
    axc.set_xticks([]); axc.set_yticks([]); axc.set_xlim(0, 1.22)
    for s in axc.spines.values():
        s.set_visible(False)

    # D - the encoder weights against the planted mixture
    axd = fig.add_subplot(gs[1, 0])
    W = np.column_stack([F[:, i_t], F[:, i_c]])
    # the planted mixture in the space the fit sees: each noise-free, unit-normed, centred electrode
    # regressed on the two planted patterns (burst in every condition; transient in audio only)
    b1 = np.tile(burst, 3); b2 = np.concatenate([trans, 0 * trans, 0 * trans])
    B = np.column_stack([b1 - b1.mean(), b2 - b2.mean()])
    Wp = np.linalg.lstsq(B, cu.reshape(N_ELEC, -1).T, rcond=None)[0].T
    Wp = Wp / np.linalg.norm(Wp, axis=0) * np.linalg.norm(W, axis=0)                   # same scale as the fitted columns
    M = np.column_stack([W, Wp])
    v = float(np.abs(M).max())
    axd.imshow(M, cmap="RdBu_r", aspect="auto", vmin=-v, vmax=v)
    axd.set_xticks(range(4)); axd.set_xticklabels(["common", "condition", "planted\ncommon", "planted\ncondition"], fontsize=7)
    axd.set_yticks(range(N_ELEC)); axd.set_yticklabels([f"e{i}" for i in range(N_ELEC)], fontsize=6.6)
    for i in range(N_ELEC):
        for j in range(4):
            axd.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6.0, color="white" if abs(M[i, j]) > 0.6 * v else INK)
    axd.axvline(1.5, color=INK, lw=0.8)
    axd.set_title("D · the weights — signed, one row per electrode\nleft: recovered · right: planted, in the same unit-normed space",
                  fontsize=9.5, loc="left", color=INK, pad=5)
    axd.tick_params(length=0)

    # E - e5 rebuilt from its two weights, per condition
    axe = fig.add_subplot(gs[1, 1])
    for c, cond in enumerate(CONDS):
        off = (2 - c) * 0.45
        axe.plot(t, xc5[c] + off, color=INK, lw=2.0)
        axe.plot(t, recon[pick, c] + off, color="#c1121f", lw=1.2, ls="--")
        axe.plot(t, F[pick, i_t] * Z[i_t, c] + off, color=COL_T, lw=1.0)
        axe.plot(t, F[pick, i_c] * Z[i_c, c] + off, color=COL_C, lw=1.0)
        axe.text(1.01, off + 0.02, cond, color=CC[cond], fontsize=7.5)
    axe.plot([], [], color=INK, lw=2, label="e5, data"); axe.plot([], [], color="#c1121f", ls="--", label="sum = reconstruction")
    axe.plot([], [], color=COL_T, label=f"{F[pick, i_t]:+.2f} × common"); axe.plot([], [], color=COL_C, label=f"{F[pick, i_c]:+.2f} × condition")
    axe.legend(fontsize=6.2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    axe.set_title("E · e5 rebuilt from its two weights\nweight × component, per condition", fontsize=9.5, loc="left", color=INK, pad=5)
    axe.set_xticks([]); axe.set_yticks([]); axe.set_xlim(0, 1.18)
    axe.spines[["top", "right"]].set_visible(False)

    # F - where the variance is, and what each component keeps in its own part
    axf = fig.add_subplot(gs[1, 2])
    labels = ["data:\ncommon part", "data:\ncondition part", "explained:\ncommon comp.", "explained:\ncondition comp."]
    vals = [var_by["time"], var_by["condition"], fit["ev"][i_t], fit["ev"][i_c]]
    cols = [COL_T, COL_C, COL_T, COL_C]
    axf.bar(range(4), [100 * x for x in vals], color=cols, width=0.62)
    for j, x in enumerate(vals):
        axf.text(j, 100 * x + 1.5, f"{100 * x:.0f} %", ha="center", fontsize=7.5, color=INK)
    axf.set_xticks(range(4)); axf.set_xticklabels(labels, fontsize=6.4)
    axf.set_ylim(0, 100); axf.set_yticks([0, 50, 100]); axf.tick_params(axis="y", labelsize=7)
    axf.text(0.02, 0.93, f"own-part share of each component: common {fit['shares'][i_t, 0]:.2f}, condition {fit['shares'][i_c, 1]:.2f}",
             transform=axf.transAxes, fontsize=6.4, color=MUTED)
    axf.set_title("F · the variance — split by the labels, then explained\nthis is the number dPCA adds", fontsize=9.5, loc="left", color=INK, pad=5)
    axf.spines[["top", "right"]].set_visible(False)

    fig.suptitle("How demixed PCA uses the condition labels — a feature set small enough to read",
                 x=0.055, y=0.972, ha="left", fontsize=15, color=INK)
    body = [
        "The same twelve invented electrodes as the convex-NMF figure, now in three conditions. Two shapes are planted: "
        "a late burst identical in every condition, and an early transient in the audio condition only. Each electrode is a "
        "known mixture of the two plus noise. Fitted with the project's own run_counter_dpca.dpca_fit.",
        "STEP ONE IS ARITHMETIC, NOT FITTING (B): the average of an electrode's three conditions is its common part; what each "
        "condition keeps after that average is subtracted is its condition part. STEP TWO finds a component inside each part (C), "
        f"with a signed weight on every electrode (D). An electrode's record is weight × component, summed (E): e5 is {F[pick, i_t]:+.2f} "
        f"of the common component and {F[pick, i_c]:+.2f} of the condition one, and the sum lands on the data (mean |error| {err:.3f}).",
        "THE COST OF THE SPLIT, shown rather than hidden: the audio-only transient is cut in three by the average - one third lands in "
        "the common part (the small early bump on the grey common line in C), two thirds stay in the condition part as +2/3 in audio "
        "and -1/3 in picture and reading. A response in one modality counts as common by a third.",
        f"WHAT IT ADDS OVER CONVEX NMF (F): the variance is split by the labels - here {100 * var_by['time']:.0f} % common, "
        f"{100 * var_by['condition']:.0f} % condition-dependent - and each component keeps {fit['shares'][i_t, 0]:.2f} / {fit['shares'][i_c, 1]:.2f} "
        "of its variance in its own part. WHAT IT LACKS: a component is a direction with signed weights, not an average of real "
        "electrodes, and nothing says which electrodes are of a type.",
    ]
    fig.text(0.055, 0.930, "\n".join(textwrap.fill(x, width=150) for x in body), fontsize=8.3, color=MUTED, va="top", linespacing=1.5)
    p = OUT / "E15_dpca_explained.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    meta = dict(n_elec=N_ELEC, n_time=N_TIME, conditions=CONDS, var_in_data=var_by, ev={"time": float(fit["ev"][i_t]), "condition": float(fit["ev"][i_c])},
                shares={"time": float(fit["shares"][i_t, 0]), "condition": float(fit["shares"][i_c, 1])}, e5_weights={"time": float(F[pick, i_t]), "condition": float(F[pick, i_c])},
                mean_abs_error=err, weights=W.tolist(), planted={"time": w_t.tolist(), "condition": w_c.tolist()})
    (OUT / "E15_dpca_explained.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"variance: common {var_by['time']:.3f}, condition {var_by['condition']:.3f}; explained {fit['ev'][i_t]:.3f} + {fit['ev'][i_c]:.3f}; mean |error| {err:.4f}")
    for i in range(N_ELEC):
        print(f"  e{i:<3} planted ({w_t[i]:.2f}, {w_c[i]:.2f})   recovered ({F[i, i_t]:+.2f}, {F[i, i_c]:+.2f})")
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
