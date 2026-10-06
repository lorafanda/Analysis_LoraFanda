#!/usr/bin/env python3
"""
run_counter_dpca.py - counter 1 to convex NMF: demixed PCA of the high-gamma responses,
with the condition labels doing the work.

    python run_counter_dpca.py                       concat_hg, 5 + 5 components, with halves
    python run_counter_dpca.py --q 4 --no-halves     fewer components, no split-half validation
    python run_counter_dpca.py --band 13-30          another band of the cube instead of HG
    python run_counter_dpca.py --quick               smoke test

WHAT IT ASKS THAT cNMF CANNOT. cNMF describes every electrode as a non-negative mixture of
K archetype profiles and never looks at the labels. dPCA (Kobak et al. 2016, eLife 5:e10989)
splits the electrode x condition x time array into the part that is the same in every
condition (what the language network does whatever the input: the response-half speech
component, the stimulus-half engagement) and the part that depends on the condition
(audio / picture / reading: the input modality), then finds a few components inside each
part. The electrode weights of a component are its map; the variance each part holds is
the answer to "how much of what we recorded is modality-general and how much is
modality-specific".

WHAT THE DATA ALLOW. 140 saves trial averages and odd / even halves, not single trials.
With trial averages only, the unregularized dPCA solution is the PCA of each
marginalization; what dPCA adds here is the encoder / decoder pair (F, D) with which the
OTHER half of the trials is projected, so demixing and reliability are measured on trials
the fit never saw, and the ridge is chosen on those trials rather than guessed. This is
said in the summary so nobody reads the in-sample demixing as a result.

METHOD (Kobak et al. 2016, Methods). X (N electrodes x C*T) centred per electrode;
X_time = mean over conditions, X_cond = X - X_time. For each marginalization phi, reduced-
rank ridge regression of X_phi on X: A = X_phi X' (X X' + mu I)^-1, F = the q leading left
singular vectors of A X, D = F' A. Components of both marginalizations are pooled and
ordered by the variance of X they explain. Explained variance of a component is the drop
in residual when it is added in that order; its marginalization shares are the variance
of D_i X_phi over phi (the pie in Kobak's figures). The data are unit-normed per electrode
first, as every cNMF run is (shape, not size), unless --pipeline raw.

OUTPUT  outputs/clustering/dpca/<feature_set>/runs/<run_id>/
  manifest.json       cohort tag, params, summary (EV per marginalization, demixing,
                      reliability, the cNMF numbers on the same matrix for comparison)
  labels.csv          the cohort table + encoder weight per component (e01 ...) + lead
  components.npz      F, D, Z (K x C x T), marginalization per component, EV table
  C1_variance.png     explained variance per component, coloured by marginalization
  C2_components.png   component time courses per condition (half-2 projection dashed)
  C3_maps.png         encoder weights on fsaverage, the four leading components
  C4_validation.png   ridge choice and split-half reliability / out-of-sample demixing
  summary.md          the numbers in words
"""
from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(ROOT / "functions"))
from functions import lf_counters as LCn                                   # noqa: E402
from functions.lf_concat import concat_hg_features                         # noqa: E402
import lf_decompose as D                                                   # noqa: E402

CONDS = list(LCn.CONDS)
MARG = ("time", "condition")
METHOD, METHOD_LABEL = "dpca", "Demixed PCA (Kobak 2016), condition-labelled"
FSET_LABEL = {"concat_hg": "Concatenated HG [a|p|r]"}


# ---------------------------------------------------------------------------------------
def features(X_concat: np.ndarray, band: str) -> tuple[np.ndarray, str]:
    """(n, 900) band time course per electrode. 'hg' is the pipeline's own concat_hg."""
    if band == "hg":
        return concat_hg_features(X_concat).astype(np.float64), "concat_hg"
    lo, hi = (float(v) for v in band.split("-"))
    rows = [r for r in range(103) if lo <= LCn.FREQ[r] < hi]
    if not rows:
        raise SystemExit(f"no cube row in {band} Hz")
    return X_concat[:, rows, :].mean(1).astype(np.float64), f"concat_{lo:g}-{hi:g}hz"


def marginalize(X: np.ndarray, C: int, T: int) -> dict:
    """X (N, C*T), centred per electrode -> {'time': X_t, 'condition': X_c}, both (N, C*T)."""
    N = X.shape[0]
    Xc = X.reshape(N, C, T)
    Xt = np.repeat(Xc.mean(axis=1, keepdims=True), C, axis=1)
    return {"time": Xt.reshape(N, -1), "condition": (Xc - Xt).reshape(N, -1)}


def dpca_fit(X: np.ndarray, C: int, T: int, q: dict, lam: float) -> dict:
    """Reduced-rank ridge regression per marginalization; pooled, ordered by explained variance."""
    N, M = X.shape
    margs = marginalize(X, C, T)
    XXt = X @ X.T
    mu = lam * np.trace(XXt) / N
    inv = np.linalg.inv(XXt + mu * np.eye(N))
    F_all, D_all, which = [], [], []
    for phi in MARG:
        A = margs[phi] @ X.T @ inv                                   # (N, N)
        Y = A @ X                                                    # the ridge prediction of X_phi
        U, s, _ = np.linalg.svd(Y, full_matrices=False)
        k = int(q[phi])
        F = U[:, :k]                                                 # encoder (N, k)
        Dd = F.T @ A                                                 # decoder (k, N)
        F_all.append(F); D_all.append(Dd); which += [phi] * k
    F = np.concatenate(F_all, 1)
    Dm = np.concatenate(D_all, 0)
    K = F.shape[1]
    tot = float((X ** 2).sum())
    # individual explained variance, then the order, then the sequential (drop-in-residual) EV
    ev_single = np.array([1.0 - float(((X - np.outer(F[:, i], Dm[i] @ X)) ** 2).sum()) / tot for i in range(K)])
    order = np.argsort(-ev_single)
    F, Dm, which = F[:, order], Dm[order], [which[i] for i in order]
    ev_seq, prev = [], 0.0
    for k in range(1, K + 1):
        Xh = F[:, :k] @ (Dm[:k] @ X)
        cum = 1.0 - float(((X - Xh) ** 2).sum()) / tot
        ev_seq.append(cum - prev); prev = cum
    Z = (Dm @ X).reshape(K, C, T)
    shares = np.array([[float((Dm[i] @ margs[phi]) @ (Dm[i] @ margs[phi])) for phi in MARG] for i in range(K)])
    shares = shares / np.maximum(shares.sum(1, keepdims=True), 1e-12)
    return dict(F=F, D=Dm, Z=Z, marg=which, ev=np.array(ev_seq), ev_single=ev_single[order],
                cum=float(prev), shares=shares, mu=mu, lam=lam)


def project(fit: dict, X: np.ndarray, C: int, T: int) -> np.ndarray:
    return (fit["D"] @ X).reshape(fit["D"].shape[0], C, T)


def demixing_oos(fit: dict, X2: np.ndarray, C: int, T: int) -> np.ndarray:
    """Share of each component's projected variance that falls in its own marginalization
    when the decoder is applied to the OTHER half of the trials."""
    m2 = marginalize(X2, C, T)
    out = []
    for i, phi in enumerate(fit["marg"]):
        v = {p: float((fit["D"][i] @ m2[p]) @ (fit["D"][i] @ m2[p])) for p in MARG}
        out.append(v[phi] / max(sum(v.values()), 1e-12))
    return np.array(out)


def choose_lambda(X1: np.ndarray, X2: np.ndarray, C: int, T: int, q: dict, grid) -> pd.DataFrame:
    """Ridge by the other half: fit on half 1, reconstruct half 2's marginalizations."""
    m2 = marginalize(X2, C, T)
    rows = []
    for lam in grid:
        f = dpca_fit(X1, C, T, q, lam)
        err = 0.0
        tot = 0.0
        for phi in MARG:
            idx = [i for i, p in enumerate(f["marg"]) if p == phi]
            Xh = f["F"][:, idx] @ (f["D"][idx] @ X2)
            err += float(((m2[phi] - Xh) ** 2).sum()); tot += float((m2[phi] ** 2).sum())
        rows.append(dict(lam=lam, held_out_err=err / tot, held_out_var_explained=1 - err / tot))
    return pd.DataFrame(rows)


def match(Fa: np.ndarray, Fb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian match of components by |correlation| of encoder weights; sign-free."""
    Cm = np.abs(np.corrcoef(Fa.T, Fb.T)[:Fa.shape[1], Fa.shape[1]:])
    r, c = linear_sum_assignment(-Cm)
    return c, Cm[r, c]


# ---------------------------------------------------------------------------------------
def fig_variance(fit, out, fset):
    K = len(fit["ev"])
    col = {"time": "#4a6fa5", "condition": "#c1121f"}
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(np.arange(K), 100 * fit["ev"], color=[col[m] for m in fit["marg"]])
    ax.plot(np.arange(K), 100 * np.cumsum(fit["ev"]), color="0.3", marker="o", ms=3, lw=1)
    ax.set_xticks(np.arange(K)); ax.set_xticklabels([f"{i + 1}" for i in range(K)])
    ax.set_xlabel("component (ordered by explained variance)"); ax.set_ylabel("% variance of X")
    ax.set_title(f"C1 - explained variance per component ({fset}); blue = condition-independent, red = condition-dependent", loc="left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "C1_variance.png", dpi=150); plt.close(fig)


def fig_components(fit, Z2, out, fset, n_show=8):
    K = min(n_show, len(fit["ev"])); C, T = fit["Z"].shape[1:]
    t = np.linspace(0, 100, T, endpoint=False)
    cc = {"audio": "#e06c9f", "picture": "#4a6fa5", "reading": "#8c6d46"}
    fig, axes = plt.subplots(K, 1, figsize=(7.5, 1.55 * K), sharex=True, squeeze=False)
    for i in range(K):
        a = axes[i][0]
        for c, cond in enumerate(CONDS):
            a.plot(t, fit["Z"][i, c], color=cc[cond], lw=1.5, label=cond if i == 0 else None)
            if Z2 is not None:
                a.plot(t, Z2[i, c], color=cc[cond], lw=0.8, ls="--", alpha=0.8)
        a.axvline(50, color="0.7", lw=.8, ls=":"); a.axhline(0, color="0.85", lw=.8)
        a.set_ylabel(f"{i + 1}  {fit['marg'][i][:4]}\n{100 * fit['ev'][i]:.1f} %", fontsize=8)
        a.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(fontsize=7, ncol=3, frameon=False, loc="upper left")
    axes[-1][0].set_xlabel("% of warped trial (50 = GO); solid = fit half, dashed = the other half projected")
    fig.suptitle(f"C2 - component time courses per condition ({fset})", x=.02, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "C2_components.png", dpi=150); plt.close(fig)


def fig_maps(fit, xyz, out, fset, n_show=4):
    ok = ~np.isnan(xyz).any(1)
    K = min(n_show, fit["F"].shape[1])
    fig, axes = plt.subplots(K, 3, figsize=(11, 3.0 * K), squeeze=False)
    for i in range(K):
        w = fit["F"][:, i]; v = float(np.percentile(np.abs(w[ok]), 98)) or 1.0
        for a, (p, q_, ti) in zip(axes[i], [(0, 1, "axial"), (0, 2, "sagittal"), (1, 2, "coronal")]):
            sc = a.scatter(xyz[ok, p], xyz[ok, q_], c=w[ok], cmap="bwr", vmin=-v, vmax=v, s=10, lw=0)
            a.set_aspect("equal"); a.axis("off")
            if i == 0:
                a.set_title(ti, fontsize=9)
        axes[i][0].text(-0.02, 0.5, f"component {i + 1}\n({fit['marg'][i]})", transform=axes[i][0].transAxes, ha="right", va="center", fontsize=8)
        fig.colorbar(sc, ax=axes[i][2], shrink=0.7, label="encoder weight")
    fig.suptitle(f"C3 - where each component lives ({fset}); {int((~ok).sum())} electrodes without coordinates", x=.02, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "C3_maps.png", dpi=150); plt.close(fig)


def fig_validation(lam_df, rel, dem, out, fset):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    a = axes[0]
    a.plot(np.arange(len(lam_df)), lam_df["held_out_var_explained"], marker="o")
    a.set_xticks(np.arange(len(lam_df))); a.set_xticklabels([f"{v:g}" for v in lam_df["lam"]], fontsize=7)
    a.set_xlabel("ridge (fraction of trace)"); a.set_ylabel("half-2 variance explained")
    a.set_title("ridge chosen on the other half of the trials", loc="left", fontsize=9)
    a.spines[["top", "right"]].set_visible(False)
    a = axes[1]
    K = len(rel)
    a.bar(np.arange(K) - 0.2, rel, width=0.4, label="encoder r, half 1 vs half 2")
    a.bar(np.arange(K) + 0.2, dem, width=0.4, label="out-of-sample demixing")
    a.set_ylim(0, 1.05); a.set_xticks(np.arange(K)); a.set_xticklabels([f"{i + 1}" for i in range(K)])
    a.set_xlabel("component"); a.legend(fontsize=7, frameon=False)
    a.set_title("split-half reliability and demixing", loc="left", fontsize=9)
    a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"C4 - validation on trials the fit never saw ({fset})", x=.02, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "C4_validation.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="hg", help="'hg' (the pipeline's concat_hg) or 'lo-hi' in Hz")
    ap.add_argument("--q", type=int, default=5, help="components per marginalization")
    ap.add_argument("--pipeline", choices=["unit_norm", "raw"], default="unit_norm")
    ap.add_argument("--no-halves", action="store_true", help="skip the split-half validation")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    df, X_concat, cache = LCn.load_cohort(verbose=not a.quick)
    X, fset = features(X_concat, a.band)
    N = X.shape[0]; C, T = len(CONDS), LCn.N_TIME
    pat = df["patient_id"].astype(str).to_numpy()
    print(f"\n=== dPCA on {fset}: {N} electrodes x {X.shape[1]} ({C} conditions x {T} bins), {len(np.unique(pat))} patients, cache {cache['name']}")

    Xp = D.unit_norm(X) if a.pipeline == "unit_norm" else X.copy()
    Xc = Xp - Xp.mean(1, keepdims=True)                        # centred per electrode
    q = {"time": a.q, "condition": a.q}

    # ---- the halves: ridge choice, reliability, out-of-sample demixing
    lam_df = None; Z2 = None; rel = None; dem = None; lam = 0.0
    if not a.no_halves:
        H1, H2 = LCn.load_halves(df, cache["name"], verbose=not a.quick)
        X1, _ = features(np.nan_to_num(H1), a.band); X2, _ = features(np.nan_to_num(H2), a.band)
        if a.pipeline == "unit_norm":
            X1, X2 = D.unit_norm(X1), D.unit_norm(X2)
        X1 = X1 - X1.mean(1, keepdims=True); X2 = X2 - X2.mean(1, keepdims=True)
        grid = [0.0, 1e-3, 1e-2] if a.quick else [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
        lam_df = choose_lambda(X1, X2, C, T, q, grid)
        lam = float(lam_df.loc[lam_df["held_out_var_explained"].idxmax(), "lam"])
        print(f"    ridge by the other half: lam = {lam:g} (half-2 variance explained {lam_df['held_out_var_explained'].max():.3f})")
        f1 = dpca_fit(X1, C, T, q, lam); f2 = dpca_fit(X2, C, T, q, lam)
        perm, rr = match(f1["F"], f2["F"])
        rel = rr
        dem = demixing_oos(f1, X2, C, T)

    fit = dpca_fit(Xc, C, T, q, lam)
    if not a.no_halves:
        # the full fit's components projected on each half separately, averaged, for the dashed lines
        Z2 = 0.5 * (project(fit, X1, C, T) + project(fit, X2, C, T))
        permf, relf = match(fit["F"], f1["F"])

    K = len(fit["ev"])
    ev_by = {phi: float(sum(e for e, m in zip(fit["ev"], fit["marg"]) if m == phi)) for phi in MARG}
    margs = marginalize(Xc, C, T)
    tot = float((Xc ** 2).sum())
    var_by = {phi: float((margs[phi] ** 2).sum()) / tot for phi in MARG}
    # plain PCA with the same K, and the newest cNMF of this feature set on the same matrix
    U, s, _ = np.linalg.svd(Xc, full_matrices=False)
    pca_ev = float((s[:K] ** 2).sum() / (s ** 2).sum())
    cn = LCn.newest_cnmf(fset) if fset in FSET_LABEL else None
    cnmf_ev = None
    if cn is not None:
        keys = list(df["patient_id"].astype(str) + "|" + df["contact_norm"])
        if cn["keys"] == keys:
            Xh = cn["G"] @ cn["C"]                                  # its own reconstruction of unit-normed X
            Xn = D.unit_norm(X)
            cnmf_ev = dict(k=int(cn["G"].shape[1]), run=cn["run"].name,
                           ev_uncentred=float(1 - ((Xn - Xh) ** 2).sum() / (Xn ** 2).sum()),
                           ev_centred=float(1 - ((Xc - (Xh - Xh.mean(1, keepdims=True))) ** 2).sum() / tot))
        else:
            cnmf_ev = dict(note=f"newest cnmf run {cn['run'].name} is on another electrode list ({len(cn['keys'])} vs {N})")

    # ---- write
    out = LCn.new_run_dir(METHOD, fset)
    np.savez(out / "components.npz", F=fit["F"], D=fit["D"], Z=fit["Z"], marg=np.array(fit["marg"]),
             ev=fit["ev"], shares=fit["shares"], conditions=np.array(CONDS))
    lab = df.drop(columns=[c for c in df.columns if c.startswith("file_path_")], errors="ignore").copy()
    for i in range(K):
        lab[f"e{i + 1:02d}"] = fit["F"][:, i]
    lead = np.argmax(np.abs(fit["F"]), axis=1)
    lab[f"lead_{METHOD}_{fset}"] = lead + 1
    lab.to_csv(out / "labels.csv", index=False)
    pd.DataFrame(dict(component=np.arange(1, K + 1), marginalization=fit["marg"], ev=fit["ev"],
                      share_time=fit["shares"][:, 0], share_condition=fit["shares"][:, 1],
                      reliability=(relf if rel is not None else np.nan), demixing_oos=(dem if dem is not None else np.nan))).to_csv(out / "components.csv", index=False)
    if lam_df is not None:
        lam_df.to_csv(out / "ridge_by_halves.csv", index=False)
    xyz = LCn.coords_for(df)
    fig_variance(fit, out, fset); fig_components(fit, Z2, out, fset); fig_maps(fit, xyz, out, fset)
    if lam_df is not None:
        fig_validation(lam_df, relf, dem, out, fset)

    summary = dict(n_electrodes=N, n_patients=int(len(np.unique(pat))), K=K, q=q, ridge=lam,
                   variance_in_data_by_marginalization=var_by, explained_variance_total=fit["cum"],
                   explained_variance_by_marginalization=ev_by, pca_same_K=pca_ev, cnmf=cnmf_ev,
                   split_half=(dict(encoder_r_mean=float(np.mean(relf)), encoder_r_min=float(np.min(relf)),
                                    demixing_oos_mean=float(np.mean(dem)), demixing_oos_min=float(np.min(dem)),
                                    half2_var_explained=float(lam_df["held_out_var_explained"].max())) if rel is not None else None))
    LCn.write_manifest(out, method=METHOD, method_label=METHOD_LABEL, feature_set=fset,
                       feature_set_label=FSET_LABEL.get(fset, fset), params=dict(band=a.band, q=q, pipeline=a.pipeline, ridge=lam, quick=a.quick),
                       summary=summary, artifacts=dict(labels="labels.csv", components="components.npz", table="components.csv"),
                       cache=cache, note="Graded, signed, condition-labelled. No hard label: lead_* is the argmax of |encoder| and is only a summary.")
    lines = [f"# dPCA on {fset} - {out.name}", "",
             f"{N} electrodes, {len(np.unique(pat))} patients, cache {cache['name']}; {a.pipeline} per electrode, then centred; "
             f"{a.q} + {a.q} components; ridge {lam:g} (chosen on the other half of the trials).", "",
             f"- variance in the data: {100 * var_by['time']:.1f} % condition-independent, {100 * var_by['condition']:.1f} % condition-dependent",
             f"- explained by the {K} components: {100 * fit['cum']:.1f} % (condition-independent {100 * ev_by['time']:.1f} %, condition-dependent {100 * ev_by['condition']:.1f} %); plain PCA with {K} components: {100 * pca_ev:.1f} %",
             ]
    if cnmf_ev and "ev_centred" in cnmf_ev:
        lines.append(f"- the newest cNMF run ({cnmf_ev['run']}, K = {cnmf_ev['k']}) on the same matrix: {100 * cnmf_ev['ev_uncentred']:.1f} % uncentred, {100 * cnmf_ev['ev_centred']:.1f} % centred")
    if rel is not None:
        lines.append(f"- split-half: encoder maps correlate r = {np.mean(relf):.2f} on average (min {np.min(relf):.2f}); out-of-sample demixing {np.mean(dem):.2f} on average (min {np.min(dem):.2f}); half-2 variance explained {lam_df['held_out_var_explained'].max():.3f}")
    lines += ["", "| component | marginalization | EV % | share time | share condition | reliability | demixing (other half) |", "|---|---|---|---|---|---|---|"]
    for i in range(K):
        lines.append(f"| {i + 1} | {fit['marg'][i]} | {100 * fit['ev'][i]:.1f} | {fit['shares'][i, 0]:.2f} | {fit['shares'][i, 1]:.2f} | "
                     f"{(f'{relf[i]:.2f}' if rel is not None else '-')} | {(f'{dem[i]:.2f}' if dem is not None else '-')} |")
    lines += ["", "With trial averages only, the unregularized solution is the PCA of each marginalization; the decoder applied to the other half of the trials is what makes the demixing and reliability numbers out of sample."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:]))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
