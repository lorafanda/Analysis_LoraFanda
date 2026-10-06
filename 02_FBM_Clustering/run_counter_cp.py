#!/usr/bin/env python3
"""
run_counter_cp.py - counter 2 to convex NMF: a CP / PARAFAC tensor decomposition of the
electrode x band x time x condition array, which cNMF flattens.

    python run_counter_cp.py                          15 bands, ranks 2..12, signed, with halves
    python run_counter_cp.py --bands 5 --rank 6       the five bands, one rank
    python run_counter_cp.py --bands rows             the cube's own rows 12-170 Hz (the resolved ones)
    python run_counter_cp.py --nonneg                 non-negative CP on power ratios 10^(dB/10)
    python run_counter_cp.py --quick                  smoke test

WHAT IT ASKS THAT cNMF CANNOT. Each electrode's record is bands x time x condition. cNMF
unrolls that into one vector and finds archetype vectors; the frequency, time and
condition structure is gone before the fit starts. CP keeps it: every component is a
spectral profile (over bands) x a time course x a condition weight x an electrode loading,
so "a high-gamma rise in the response half, in every condition, on these electrodes" is one
component with four readable parts, and "a beta drop in the stimulus half, picture only" is
another. The condition weights are where the labels enter. Miwakeichi et al. 2004
(NeuroImage 22:1035) and Mørup et al. 2006 (NeuroImage 29:938) did this on channel x
frequency x time EEG; Williams et al. 2018 (Neuron 98:1099) on neuron x time x trial.

METHOD. Alternating least squares with a small ridge (Kolda & Bader 2009), several random
starts, the best fit kept; --nonneg switches to multiplicative updates on 10^(dB/10), which
is non-negative (dB is not). Factors are unit-normed, the scale sits in the component
weights. Rank is read from three curves written for ranks 2..R: variance explained,
core consistency (CORCONDIA, Bro & Kiers 2003; reported, not used - three conditions
leave its core underdetermined from rank 3 on) and split-half similarity (the halves of
the trials fitted separately, components matched by the Hungarian method on the product of
their factor congruences, Williams et al. 2018's similarity score). The data are
unit-normed per electrode first, as every cNMF run is, unless --pipeline raw.

OUTPUT  outputs/clustering/cp/<feature_set>/runs/<run_id>/
  manifest.json      cohort tag, params, summary (EV, CORCONDIA, split-half similarity)
  labels.csv         the cohort table + electrode loading per component (a01 ...) + lead
  factors.npz        A (electrodes), B (bands), Tm (time), Cn (conditions), weights
  rank_curves.csv    EV / CORCONDIA / similarity per rank
  C1_rank.png        the three curves
  C2_components.png  per component: band profile, time course per condition, weights
  C3_maps.png        electrode loadings on fsaverage, the four leading components
  summary.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from functions import lf_counters as LCn                                                     # noqa: E402
from functions.lf_features import FREQ_BANDS_15_TO_400HZ, FREQ_BANDS_5_TO_400HZ             # noqa: E402

CONDS = list(LCn.CONDS)
METHOD, METHOD_LABEL = "cp", "CP / PARAFAC (electrode x band x time x condition)"


# ---------------------------------------------------------------------------------------
def band_set(name: str):
    if name == "15":
        return list(FREQ_BANDS_15_TO_400HZ), "bands15"
    if name == "5":
        return list(FREQ_BANDS_5_TO_400HZ), "bands5"
    if name == "rows":
        rows = [r for r in range(103) if 12 <= LCn.FREQ[r] < 170]
        return [(LCn.FREQ[r], LCn.FREQ[r] + LCn.DF_HZ) for r in rows], "rows12-170"
    raise SystemExit(f"--bands must be 15, 5 or rows (got {name})")


def khatri_rao(mats):
    out = mats[0]
    for M in mats[1:]:
        out = (out[:, None, :] * M[None, :, :]).reshape(-1, out.shape[1])
    return out


def unfold(Y, mode):
    return np.moveaxis(Y, mode, 0).reshape(Y.shape[mode], -1)


def recon(factors, w):
    """Full tensor from unit-norm factors and weights (einsum over the four modes)."""
    A, B, Tm, Cn = factors
    return np.einsum("ir,jr,kr,lr,r->ijkl", A, B, Tm, Cn, w, optimize=True)


def cp_als(Y, R, *, n_iter=300, tol=1e-6, nonneg=False, seed=0, ridge=1e-6, verbose=False):
    """CP of a 4-way tensor. Returns (factors unit-normed, weights, variance explained)."""
    rng = np.random.default_rng(seed)
    dims = Y.shape
    F = [np.abs(rng.standard_normal((d, R))) if nonneg else rng.standard_normal((d, R)) for d in dims]
    F = [f / np.linalg.norm(f, axis=0, keepdims=True) for f in F]
    tot = float((Y ** 2).sum())
    prev = np.inf
    for it in range(n_iter):
        for n in range(4):
            others = [F[m] for m in range(4) if m != n]
            V = np.ones((R, R))
            for M in others:
                V *= M.T @ M
            KR = khatri_rao(others)                             # rows ordered like the unfolding: the last remaining mode varies fastest
            MTTKRP = unfold(Y, n) @ KR
            if nonneg:
                F[n] = F[n] * MTTKRP / np.maximum(F[n] @ V, 1e-12)
            else:
                F[n] = np.linalg.solve(V + ridge * np.eye(R), MTTKRP.T).T
            if n < 3:                                           # keep the scale in the last mode during the sweep
                nr = np.linalg.norm(F[n], axis=0, keepdims=True); nr[nr == 0] = 1
                F[n] = F[n] / nr
        if it % 10 == 0 or it == n_iter - 1:
            w = np.linalg.norm(F[3], axis=0)
            fac = [F[0], F[1], F[2], F[3] / np.where(w == 0, 1, w)]
            err = float(((Y - recon(fac, w)) ** 2).sum()) / tot
            if verbose:
                print(f"      iter {it:4d}  rel err {err:.5f}")
            if abs(prev - err) < tol:
                break
            prev = err
    w = np.linalg.norm(F[3], axis=0)
    fac = [F[0], F[1], F[2], F[3] / np.where(w == 0, 1, w)]
    order = np.argsort(-w)
    fac = [f[:, order] for f in fac]; w = w[order]
    ev = 1.0 - float(((Y - recon(fac, w)) ** 2).sum()) / tot
    return fac, w, ev


def cp_best(Y, R, *, n_init=4, **kw):
    best = None
    for s in range(n_init):
        fac, w, ev = cp_als(Y, R, seed=s, **kw)
        if best is None or ev > best[2]:
            best = (fac, w, ev)
    return best


def corcondia(Y, factors, w) -> float:
    """Core consistency: the least-squares Tucker core of the fitted factors against the
    superdiagonal (Bro & Kiers 2003). 100 = perfectly trilinear, < ~50 = too many components."""
    A, B, Tm, Cn = factors
    A = A * w                                                   # put the weights on one mode
    G = Y
    for n, M in enumerate((A, B, Tm, Cn)):
        P = np.linalg.pinv(M)                                   # (R, d_n)
        G = np.moveaxis(np.tensordot(P, G, axes=(1, n)), 0, n)
    R = A.shape[1]
    I = np.zeros((R, R, R, R))
    for r in range(R):
        I[r, r, r, r] = 1.0
    return float(100 * (1 - ((G - I) ** 2).sum() / R))


def congruence(fa, fb) -> np.ndarray:
    """|Tucker congruence| matrix between the columns of two factor matrices."""
    na = fa / np.maximum(np.linalg.norm(fa, axis=0, keepdims=True), 1e-12)
    nb = fb / np.maximum(np.linalg.norm(fb, axis=0, keepdims=True), 1e-12)
    return np.abs(na.T @ nb)


def similarity(f1, f2) -> tuple[float, np.ndarray]:
    """Williams 2018: match components across two fits; score = product of the four modes'
    congruences, averaged over the matched pairs."""
    S = np.ones((f1[0].shape[1], f2[0].shape[1]))
    for a, b in zip(f1, f2):
        S *= congruence(a, b)
    r, c = linear_sum_assignment(-S)
    return float(S[r, c].mean()), c


# ---------------------------------------------------------------------------------------
def fig_rank(curves: pd.DataFrame, out, fset):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for a, col, lab in zip(axes, ("ev", "corcondia", "similarity"), ("variance explained", "core consistency (%)", "split-half similarity")):
        if curves[col].notna().any():
            a.plot(curves["rank"], curves[col], marker="o")
        a.set_xlabel("components"); a.set_title(lab, loc="left", fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"C1 - choosing the rank ({fset})", x=.02, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "C1_rank.png", dpi=150); plt.close(fig)


def fig_components(fac, w, bands, out, fset, n_show=8):
    A, B, Tm, Cn = fac
    R = min(n_show, A.shape[1]); T = Tm.shape[0]
    t = np.linspace(0, 100, T, endpoint=False)
    cc = {"audio": "#e06c9f", "picture": "#4a6fa5", "reading": "#8c6d46"}
    labels = [f"{lo:g}" for lo, _ in bands]
    fig, axes = plt.subplots(R, 3, figsize=(12, 1.7 * R), squeeze=False, gridspec_kw=dict(width_ratios=[1.2, 2.2, 0.8]))
    for r in range(R):
        a = axes[r][0]
        a.bar(np.arange(len(bands)), B[:, r], color="0.35")
        a.axhline(0, color="0.8", lw=.8)
        if r == R - 1:
            a.set_xticks(np.arange(len(bands))); a.set_xticklabels(labels, fontsize=5, rotation=90); a.set_xlabel("band (lower edge, Hz)")
        else:
            a.set_xticks([])
        a.set_ylabel(f"comp {r + 1}\nw = {w[r]:.2f}", fontsize=8)
        a = axes[r][1]
        for c, cond in enumerate(CONDS):
            a.plot(t, Tm[:, r] * Cn[c, r], color=cc[cond], lw=1.4, label=cond if r == 0 else None)
        a.axvline(50, color="0.7", lw=.8, ls=":"); a.axhline(0, color="0.85", lw=.8)
        if r == R - 1:
            a.set_xlabel("% of warped trial (50 = GO)")
        a = axes[r][2]
        a.bar(np.arange(3), Cn[:, r], color=[cc[c] for c in CONDS]); a.axhline(0, color="0.8", lw=.8)
        a.set_xticks(np.arange(3)); a.set_xticklabels(CONDS, fontsize=7)
        for ax in axes[r]:
            ax.spines[["top", "right"]].set_visible(False)
    axes[0][1].legend(fontsize=7, ncol=3, frameon=False, loc="upper left")
    axes[0][0].set_title("band profile", loc="left", fontsize=9); axes[0][1].set_title("time course x condition weight", loc="left", fontsize=9); axes[0][2].set_title("condition weights", loc="left", fontsize=9)
    fig.suptitle(f"C2 - the components ({fset})", x=.02, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "C2_components.png", dpi=150); plt.close(fig)


def fig_maps(A, xyz, out, fset, n_show=4):
    ok = ~np.isnan(xyz).any(1)
    R = min(n_show, A.shape[1])
    fig, axes = plt.subplots(R, 3, figsize=(11, 3.0 * R), squeeze=False)
    for r in range(R):
        v = float(np.percentile(np.abs(A[ok, r]), 98)) or 1.0
        for a, (p, q_, ti) in zip(axes[r], [(0, 1, "axial"), (0, 2, "sagittal"), (1, 2, "coronal")]):
            sc = a.scatter(xyz[ok, p], xyz[ok, q_], c=A[ok, r], cmap="bwr", vmin=-v, vmax=v, s=10, lw=0)
            a.set_aspect("equal"); a.axis("off")
            if r == 0:
                a.set_title(ti, fontsize=9)
        axes[r][0].text(-0.02, 0.5, f"component {r + 1}", transform=axes[r][0].transAxes, ha="right", va="center", fontsize=8)
        fig.colorbar(sc, ax=axes[r][2], shrink=0.7, label="electrode loading")
    fig.suptitle(f"C3 - where each component lives ({fset}); {int((~ok).sum())} electrodes without coordinates", x=.02, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "C3_maps.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", default="15", help="15 | 5 | rows (the cube's rows 12-170 Hz)")
    ap.add_argument("--time-bins", type=int, default=100)
    ap.add_argument("--rank", type=int, default=None, help="fit this rank only (the curves still run over --ranks)")
    ap.add_argument("--ranks", default="2-12")
    ap.add_argument("--n-init", type=int, default=4)
    ap.add_argument("--nonneg", action="store_true", help="non-negative CP on 10^(dB/10)")
    ap.add_argument("--pipeline", choices=["unit_norm", "raw"], default="unit_norm")
    ap.add_argument("--no-halves", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    bands, bname = band_set(a.bands)
    fset = f"concat_{bname}" + ("_ratio" if a.nonneg else "")
    df, X_concat, cache = LCn.load_cohort(verbose=not a.quick)
    pat = df["patient_id"].astype(str).to_numpy()
    tb = 50 if a.quick else a.time_bins
    Y = LCn.band_tensor(X_concat, bands, time_bins=tb)
    if a.nonneg:
        Y = 10 ** (Y / 10)
    if a.pipeline == "unit_norm":
        Y = LCn.unit_norm_slices(Y)
    Y = Y.astype(np.float64)
    print(f"\n=== CP on {fset}: tensor {Y.shape} (electrodes x bands x time x conditions), {len(np.unique(pat))} patients, cache {cache['name']}")
    lo, hi = (int(v) for v in a.ranks.split("-"))
    ranks = [2, 3, 4] if a.quick else list(range(lo, hi + 1))
    n_init = 2 if a.quick else a.n_init
    n_iter = 60 if a.quick else 300

    H1 = H2 = None
    if not a.no_halves:
        X1, X2 = LCn.load_halves(df, cache["name"], verbose=not a.quick)
        H1 = LCn.band_tensor(np.nan_to_num(X1), bands, time_bins=tb); H2 = LCn.band_tensor(np.nan_to_num(X2), bands, time_bins=tb)
        if a.nonneg:
            H1, H2 = 10 ** (H1 / 10), 10 ** (H2 / 10)
        if a.pipeline == "unit_norm":
            H1, H2 = LCn.unit_norm_slices(H1), LCn.unit_norm_slices(H2)
        H1, H2 = H1.astype(np.float64), H2.astype(np.float64)

    rows = []
    fits = {}
    for R in ranks:
        fac, w, ev = cp_best(Y, R, n_init=n_init, n_iter=n_iter, nonneg=a.nonneg)
        cc = corcondia(Y, fac, w)
        sim = np.nan
        if H1 is not None:
            f1, w1, _ = cp_best(H1, R, n_init=n_init, n_iter=n_iter, nonneg=a.nonneg)
            f2, w2, _ = cp_best(H2, R, n_init=n_init, n_iter=n_iter, nonneg=a.nonneg)
            sim, _ = similarity(f1, f2)
        rows.append(dict(rank=R, ev=ev, corcondia=cc, similarity=sim))
        fits[R] = (fac, w, ev)
        print(f"    rank {R:2d}: variance explained {ev:.3f}   core consistency {cc:6.1f}   split-half similarity {sim:.3f}")
    curves = pd.DataFrame(rows)
    # THE RANK RULE: the largest rank whose split-half similarity is >= 0.85; if none reaches
    # it, the rank with the best similarity. Core consistency is written but not used to
    # decide: the condition mode has three levels, so from rank 3 on the least-squares core
    # behind CORCONDIA is underdetermined and the number is meaningless on this tensor.
    if a.rank:
        R = int(a.rank)
    else:
        sim = curves["similarity"].fillna(-1.0)
        good = curves.loc[sim >= 0.85, "rank"]
        R = int(good.max()) if len(good) else int(curves.loc[sim.idxmax(), "rank"])
    if R not in fits:
        fits[R] = cp_best(Y, R, n_init=n_init, n_iter=n_iter, nonneg=a.nonneg)
    fac, w, ev = fits[R]

    out = LCn.new_run_dir(METHOD, fset)
    curves.to_csv(out / "rank_curves.csv", index=False)
    np.savez(out / "factors.npz", A=fac[0], B=fac[1], Tm=fac[2], Cn=fac[3], weights=w,
             bands=np.array(bands), conditions=np.array(CONDS))
    lab = df.drop(columns=[c for c in df.columns if c.startswith("file_path_")], errors="ignore").copy()
    for r in range(R):
        lab[f"a{r + 1:02d}"] = fac[0][:, r]
    lab[f"lead_{METHOD}_{fset}"] = np.argmax(np.abs(fac[0]) * w, axis=1) + 1
    lab.to_csv(out / "labels.csv", index=False)
    xyz = LCn.coords_for(df)
    fig_rank(curves, out, fset); fig_components(fac, w, bands, out, fset); fig_maps(fac[0], xyz, out, fset)
    row = curves[curves["rank"] == R].iloc[0] if (curves["rank"] == R).any() else None
    summary = dict(n_electrodes=int(Y.shape[0]), n_patients=int(len(np.unique(pat))), tensor=list(Y.shape), rank=R,
                   variance_explained=ev, corcondia=(float(row["corcondia"]) if row is not None else None),
                   split_half_similarity=(float(row["similarity"]) if row is not None and np.isfinite(row["similarity"]) else None),
                   rank_curves=curves.to_dict("records"), weights=[float(x) for x in w])
    LCn.write_manifest(out, method=METHOD, method_label=METHOD_LABEL, feature_set=fset, feature_set_label=f"CP tensor, {bname}, {tb} bins",
                       params=dict(bands=a.bands, time_bins=tb, rank=R, ranks=ranks, n_init=n_init, n_iter=n_iter, nonneg=a.nonneg, pipeline=a.pipeline, quick=a.quick),
                       summary=summary, artifacts=dict(labels="labels.csv", factors="factors.npz", curves="rank_curves.csv"), cache=cache,
                       note="Graded, signed. No hard label: lead_* is the argmax of |loading| x weight and is only a summary.")
    lines = [f"# CP on {fset} - {out.name}", "",
             f"tensor {Y.shape} (electrodes x bands x time x conditions), {len(np.unique(pat))} patients, cache {cache['name']}; {a.pipeline} per electrode; "
             f"{'non-negative on power ratios' if a.nonneg else 'signed, on dB'}; {n_init} starts, {n_iter} iterations.", "",
             "| rank | variance explained | core consistency | split-half similarity |", "|---|---|---|---|"]
    for _, r_ in curves.iterrows():
        sim_txt = "-" if not np.isfinite(r_["similarity"]) else f"{r_['similarity']:.3f}"
        lines.append(f"| {int(r_['rank'])} | {r_['ev']:.3f} | {r_['corcondia']:.1f} | {sim_txt} |")
    lines += ["", f"chosen rank {R}: variance explained {ev:.3f}; weights " + ", ".join(f"{x:.2f}" for x in w),
              "", "Rank rule: the largest rank whose split-half similarity is >= 0.85, else the most similar one (override with --rank). "
              "Core consistency is reported only: with three conditions the core behind it is underdetermined from rank 3 on."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:]))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
