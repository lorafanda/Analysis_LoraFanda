#!/usr/bin/env python3
"""
run_decomposition.py - convex-NMF decomposition, one folder per FEATURE SET.

Why this exists as a script and not only as notebook cells:

  * 235/236 each hard-coded a single `RUN` and wrote to one shared
    `outputs/clustering/decomposition/` with fixed filenames. Running a second
    feature set silently overwrote the first, and nothing recorded which run had
    produced the files sitting there. On 2026-08-17 that produced a folder where
    F3a/F3b/F4/F5 described the new 1266-electrode cohort while F7 described the
    old 1027-electrode one.
  * The rank curve and the leave-one-patient-out tables were READ by
    make_decomposition_figure.py and make_overview_options.py but written by
    nothing, so D3_graded_decomposition.png froze at its 2026-08-08 inputs.

Both are fixed here: output is `decomposition/<feature_set>/`, every intermediate
the downstream scripts read is written, and meta.json records exactly which source
run, which K and which cohort produced it.

ONE DECOMPOSITION PER FEATURE SET, NOT PER RUN. X_train.npy is byte-identical
between the k-means and hierarchical runs of the same feature set (verified by
md5) - the clustering algorithm produces labels, it does not touch the features,
and convex NMF consumes only the features. So k-means/hierarchical is not an axis
of comparison here; concat_hg vs concat_rawds is.

    python run_decomposition.py                      # both feature sets
    python run_decomposition.py --feature-set concat_hg
    python run_decomposition.py --quick              # small sweeps, for smoke-testing
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))
import lf_decompose as D  # noqa: E402

CLUST = ROOT / "outputs" / "clustering"
DEC = CLUST / "decomposition"
COORDS = ROOT / "outputs" / "250_recon" / "fsaverage" / "coords" / "ALL_PATIENTS_contacts_fsaverage.csv"

# Source of X for each feature set. The k-means run is used purely as the carrier
# of X_train.npy and the electrode order; its cluster labels are not read.
SOURCES = {
    "concat_hg":    CLUST / "kmeans" / "concat_hg" / "runs" / "20260817_171544",
    "concat_rawds": CLUST / "kmeans" / "concat_rawds" / "runs" / "20260817_171634",
}
# K per feature set. 7 for concat_hg is the established choice (knee of the
# held-out curve is 5->6; 7 was validated in 236). concat_rawds gets its own
# entry because it is a different feature space - do not inherit it silently.
KS_DEFAULT = {"concat_hg": 7, "concat_rawds": 7, "concat_hg_all": 7}

# concat_hg_all has no run id until 237 has been run, so pinning it here would mean
# editing this file after every rebuild - the drift lf_runs exists to prevent.
FEATURE_SETS = sorted(set(SOURCES) | {"concat_hg_all"})


def source_for(fset):
    """The k-means run whose X_train the decomposition is fitted on."""
    p = SOURCES.get(fset)
    if p is not None:
        return p
    sys.path.insert(0, str(ROOT / "functions"))
    import lf_runs as LR
    return LR.newest_run("kmeans", fset)


def parse_schema(src: Path):
    """Return (conditions, bands, n_time) from feature_schema.json.

    concat_hg    -> ['audio','picture','reading'], ['hg'], 300
    concat_rawds -> same conditions, 15 bands, 30 time bins

    Derived rather than assumed: the two feature sets are shaped differently, and
    plotting one as if it were the other is exactly the bug this replaces.
    """
    names = json.loads((src / "feature_schema.json").read_text())["feature_names"]
    conds, bands, times = [], [], set()
    for f in names:
        c, b, t = f.split("|")
        if c not in conds:
            conds.append(c)
        if b not in bands:
            bands.append(b)
        times.add(t)
    return conds, bands, len(times)


def load(fset: str):
    src = source_for(fset)
    X = np.load(src / "X_train.npy").astype(np.float64)
    lab = pd.read_csv(src / "labels.csv")
    conds, bands, nt = parse_schema(src)
    assert X.shape[1] == len(conds) * len(bands) * nt, (
        f"{fset}: schema says {len(conds)}x{len(bands)}x{nt} "
        f"but X has {X.shape[1]} columns")

    co = pd.read_csv(COORDS)

    def nz(s):
        return str(s).replace("_", "").replace("-", "").upper()

    co["key"] = [f"{p}|{nz(x)}" for p, x in zip(co["patient"], co["name"])]
    lab["key"] = [f"{p}|{nz(e)}" for p, e in zip(lab["patient_id"], lab["electrode"])]
    xyz = lab.merge(co[["key", "x", "y", "z"]], on="key", how="left")[["x", "y", "z"]].to_numpy()
    return src, X, lab, xyz, conds, bands, nt


def plot_components(C, conds, bands, nt, K, out, fset):
    """One row per component. A single band is a time course; many bands are a
    band x time image. Plotting a 15-band block on a time axis is meaningless."""
    if len(bands) == 1:
        t = np.linspace(0, 100, nt)
        fig, axes = plt.subplots(K, len(conds), figsize=(12, 2.1 * K),
                                 sharex=True, sharey=True, squeeze=False)
        for j in range(K):
            for b, cond in enumerate(conds):
                a = axes[j][b]
                a.plot(t, C[j].reshape(len(conds), nt)[b], lw=1.6, color=f"C{j % 10}")
                a.axvline(50, color="0.7", lw=.8, ls=":")   # GO cue
                a.axhline(0, color="0.85", lw=.8)
                if j == 0:
                    a.set_title(cond, fontsize=10)
                if b == 0:
                    a.set_ylabel(f"component {j}", fontsize=9)
                a.spines[["top", "right"]].set_visible(False)
        axes[-1][len(conds) // 2].set_xlabel("% of warped trial (50 = GO cue)")
    else:
        v = float(np.abs(C).max())
        fig, axes = plt.subplots(K, len(conds), figsize=(4.0 * len(conds), 1.9 * K),
                                 sharex=True, sharey=True, squeeze=False)
        for j in range(K):
            blk = C[j].reshape(len(conds), len(bands), nt)
            for b, cond in enumerate(conds):
                a = axes[j][b]
                a.imshow(blk[b], aspect="auto", origin="lower", cmap="bwr",
                         vmin=-v, vmax=v, extent=[0, 100, 0, len(bands)],
                         interpolation="nearest")
                a.axvline(50, color="0.35", lw=.8, ls=":")
                if j == 0:
                    a.set_title(cond, fontsize=10)
                if b == 0:
                    a.set_ylabel(f"comp {j}", fontsize=9)
                    a.set_yticks(np.arange(len(bands)) + .5)
                    a.set_yticklabels(bands, fontsize=5)
                else:
                    a.set_yticks([])
        axes[-1][len(conds) // 2].set_xlabel("% of warped trial (50 = GO cue)")
    fig.suptitle(f"F3b - component profiles ({fset})", x=.09, ha="left")
    fig.tight_layout()
    fig.savefig(out / "F3b_components.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(fset: str, k, quick: bool) -> dict:
    out = DEC / fset
    out.mkdir(parents=True, exist_ok=True)
    src, X, lab, xyz, conds, bands, nt = load(fset)
    pat = lab["patient_id"].astype(str).to_numpy()
    print(f"\n=== {fset}: {X.shape[0]} electrodes x {X.shape[1]} features | "
          f"{len(np.unique(pat))} patients | "
          f"{int(np.isnan(xyz).any(1).sum())} without coordinates")
    print(f"    structure: {len(conds)} conditions x {len(bands)} band(s) x {nt} time bins")
    print(f"    source: {src.relative_to(ROOT)}")

    Xs = D.unit_norm(X)

    # ---- 1. rank curve. Written to CSV as well as PNG: make_decomposition_figure.py
    #         and make_overview_options.py read the CSV, and it had gone stale.
    KS = [2, 3, 4, 5] if quick else [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 19, 20, 21, 22, 23]
    cv = D.cv_rank_curve(Xs, KS, n_folds=5, n_iter=40 if quick else 150)
    cv.to_csv(out / "cv_rank_curve.csv", index=False)
    g = cv.groupby("k")["var_explained"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=3)
    ax.set_xlabel("components")
    ax.set_ylabel("held-out variance explained")
    ax.set_title(f"F3a - rank chosen on data the fit never saw ({fset})", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "F3a_cv_rank.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    K = int(k or KS_DEFAULT[fset])
    W, G, C = D.convex_nmf(Xs, K, random_state=0, n_iter=60 if quick else 300)
    ve = float(1 - ((Xs - D.reconstruct(Xs, W, G)) ** 2).sum() / (Xs ** 2).sum())
    print(f"    K={K}  in-sample variance explained: {ve:.3f}")

    plot_components(C, conds, bands, nt, K, out, fset)

    # ---- 2. how graded is it
    Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
    mix = D.mixture_summary(Gn)
    top = Gn.max(1)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(top, bins=40, color="#4a6fa5")
    for v, lb in ((0.5, "no majority"), (0.8, "dominated")):
        ax.axvline(v, color="#c1121f", ls="--", lw=1.2)
        ax.text(v, ax.get_ylim()[1] * .92, lb, fontsize=8, rotation=90, ha="right")
    ax.set_xlabel("largest component weight per electrode")
    ax.set_ylabel("electrodes")
    ax.set_title(f"F4 - how much does the leading component lead? ({fset})", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "F4_mixture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 3. anatomy, hue = lead, saturation = confidence
    ok = ~np.isnan(xyz).any(1)
    rgb = D.soft_rgb(Gn, plt.get_cmap("tab20").colors[:K])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for a, (i, j, ti) in zip(axes, [(0, 1, "axial (x, y)"), (0, 2, "sagittal (x, z)"),
                                    (1, 2, "coronal (y, z)")]):
        a.scatter(xyz[ok, i], xyz[ok, j], c=rgb[ok], s=16, lw=0)
        a.set_title(ti, fontsize=10)
        a.set_aspect("equal")
        a.axis("off")
    fig.suptitle(f"F5 - leading component (hue), confidence (saturation) ({fset})",
                 x=.02, ha="left")
    fig.tight_layout()
    fig.savefig(out / "F5_soft_anatomy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    coh_obs, coh_ratio = D.spatial_coherence(Gn.argmax(1), xyz)

    # ---- 4. which components share cortex
    names = [f"c{j}" for j in range(K)]
    Cm = pd.DataFrame(np.corrcoef(Gn[ok].T), index=names, columns=names)
    Cm.to_csv(out / "component_correlation.csv")

    # ---- 5. leave-one-patient-out against a size-matched null. Both tables are
    #         written: an ARI minimum means nothing without the null beside it.
    real = D.lopo_stability(Xs, pat, K)
    null = D.pseudo_group_null(Xs, pat, K, n_rep=2 if quick else 6)
    real.to_csv(out / "lopo_patients.csv", index=False)
    null.to_csv(out / "lopo_null.csv", index=False)
    okf = ~real["degenerate"]
    z = float((real.loc[okf, "ari"].min() - null["min"].mean()) / max(null["min"].std(), 1e-9))

    # ---- 6. consensus across preprocessing pipelines
    Cass = D.pipeline_consensus(X, K, patient=pat)
    ycon = D.consensus_labels(Cass, K)
    con_obs, con_ratio = D.spatial_coherence(ycon, xyz)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    o = np.argsort(ycon)
    im = ax.imshow(Cass[np.ix_(o, o)], cmap="magma", vmin=0, vmax=1)
    ax.set_title(f"F7 - co-association across pipelines x seeds ({fset})",
                 loc="left", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, label="fraction co-clustered")
    fig.savefig(out / "F7_consensus.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    np.save(out / "consensus_matrix.npy", Cass)

    # ---- 7. discovery / replication split over patient halves
    rng = np.random.default_rng(0)
    ps = np.unique(pat)
    rows = []
    for rep in range(3 if quick else 10):
        sh = rng.permutation(ps)
        A = set(sh[:len(ps) // 2])
        ma = np.isin(pat, list(A))
        _, _, Ca = D.convex_nmf(Xs[ma], K, random_state=0, n_iter=40 if quick else 150)
        _, _, Cb = D.convex_nmf(Xs[~ma], K, random_state=0, n_iter=40 if quick else 150)
        M = np.corrcoef(Ca, Cb)[:K, K:]
        rows.append(dict(rep=rep, mean_best_match=float(M.max(1).mean())))
    rep_df = pd.DataFrame(rows)
    rep_df.to_csv(out / "replication_halves.csv", index=False)

    np.save(out / "G_loadings.npy", G)
    np.save(out / "components.npy", C)
    lab.assign(**{f"w{j}": Gn[:, j] for j in range(K)}).to_csv(
        out / "electrode_loadings.csv", index=False)

    meta = dict(
        feature_set=fset, K=K,
        source_run=str(src.relative_to(ROOT)).replace("\\", "/"),
        n_electrodes=int(X.shape[0]), n_features=int(X.shape[1]),
        n_patients=int(len(np.unique(pat))),
        conditions=conds, bands=bands, n_time=nt,
        in_sample_var_explained=ve,
        mixture=mix,
        argmax_spatial_coherence=[float(coh_obs), float(coh_ratio)],
        consensus_spatial_coherence=[float(con_obs), float(con_ratio)],
        consensus_sizes=[int(x) for x in np.bincount(ycon, minlength=K)],
        lopo_real_mean=float(real.loc[okf, "ari"].mean()),
        lopo_real_min=float(real.loc[okf, "ari"].min()),
        lopo_degenerate_folds=int((~okf).sum()),
        lopo_null_min_mean=float(null["min"].mean()),
        lopo_null_min_std=float(null["min"].std()),
        lopo_z=z,
        replication_mean=float(rep_df["mean_best_match"].mean()),
        quick=bool(quick),
        written=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"    graded: {json.dumps(mix)}")
    print(f"    LOPO min {meta['lopo_real_min']:.3f} vs null "
          f"{meta['lopo_null_min_mean']:.3f} +/- {meta['lopo_null_min_std']:.3f}"
          f"  (z = {z:+.2f})")
    print(f"    replication (best-match r across patient halves): "
          f"{meta['replication_mean']:.3f}")
    print(f"    -> {out}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=FEATURE_SETS, action="append")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    sets = a.feature_set or sorted(SOURCES)   # the default stays the two published sets
    metas = {}
    for fs in sets:
        metas[fs] = run(fs, a.k, a.quick)
    idx = DEC / "index.json"
    prev = {}
    if idx.exists():
        try:
            prev = json.loads(idx.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    prev.update(metas)
    idx.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    print(f"\nwrote {idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
