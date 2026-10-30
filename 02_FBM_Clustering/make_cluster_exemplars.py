#!/usr/bin/env python3
"""The most representative electrode of every cluster, shown as its RAW ERSP.

    python make_cluster_exemplars.py            # every cell: 4 feature sets x 3 algorithms
    python make_cluster_exemplars.py --feature-set concat_bands15 --method cnmf
    python make_cluster_exemplars.py --top 5    # more candidates in the table (figure shows the first)

REPRESENTATIVE = the member electrode closest to its cluster's centroid in the space the
clustering was fitted in (unit-norm rows of the feature set): cosine similarity between the
electrode's row and the mean of its cluster's rows. The same rule for every algorithm, so
the pages compare; for convex NMF the electrode's own-cluster loading is reported beside it.

One page per cell, one row per cluster:
    left    the cluster's mean in the feature space (what FIG 1 draws)
    middle  the exemplar electrode in the same feature space (what the clustering saw)
    right   the exemplar's RAW ERSP: the three conditions' 0-400 Hz x time-normalised
            matrices from 03_ERSP (the files named in the run's labels.csv), +-7 dB, the GO
            cue dashed at mid-block, exactly as FIG 0 draws its four examples.

OUTPUTS  outputs/clustering/exemplars/<cache>/
    exemplars_<fset>_<method>_K<k>_run<id>.png
    exemplars_<fset>_<method>_K<k>_run<id>.csv     the top candidates per cluster, with the
                                                   three raw-cube paths of each
    exemplars_index.csv                            one line per cluster over every cell
Nothing published is touched.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("p2fig1", ROOT / "00_Paper2_Figures.py")
P2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P2)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))
import measure_cluster_stability as MS                    # noqa: E402
import lf_runs as LR                                      # noqa: E402

_dspec = importlib.util.spec_from_file_location("p2driver", ROOT / "run_paper_figures_v17.py")
DRV = importlib.util.module_from_spec(_dspec)
_dspec.loader.exec_module(DRV)
K_BY = DRV.K

OUT = ROOT / "outputs" / "clustering" / "exemplars"
INK, MUTED, GREY = P2.INK, P2.MUTED, P2.GREY
VLIM, FMAX = 7.0, 400.0                      # FIG 0's raw-ERSP scale and frequency range
METHODS = ["cnmf", "kmeans", "hierarchical"]
FS_SHORT = {"concat_bands15": "15 bands", "concat_bands5": "5 bands",
            "concat_bands2": "2 bands", "concat_hfa": "HFA"}
CONDS = ["audio", "picture", "reading"]


def region_table():
    """patient|norm(electrode) -> an anatomy label, from the recon coordinate table
    (HUG anatomy sits in `label`, Bern in `aparc`); empty when neither is there."""
    co = pd.read_csv(P2.COORDS)
    cols = [c for c in ("label", "aparc", "region", "anat") if c in co.columns]
    key = [f"{p}|{P2.norm(n)}" for p, n in zip(co["patient"], co["name"])]
    if not cols:
        return {}
    out = {}
    for k, row in zip(key, co[cols].itertuples(index=False)):
        vals = [str(v) for v in row if isinstance(v, str) and v and v.lower() != "nan"]
        if vals and k not in out:
            out[k] = vals[0]
    return out


def candidates(d, top):
    """Per cluster, the `top` members closest to the centroid (cosine, unit-norm rows)."""
    A = MS.unit(d["X"])
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    own = d["Gn"].max(1)
    rows = []
    for j in range(d["k"]):
        m = np.where(d["lab"] == j)[0]
        c = An[m].mean(0)
        c = c / max(np.linalg.norm(c), 1e-12)
        cos = An[m] @ c
        order = m[np.argsort(-cos)]
        for r, i in enumerate(order[:top], 1):
            rows.append(dict(cluster=j, rank=r, idx=int(i), cosine=float(An[i] @ c),
                             loading=float(own[i]) if not d["hard"] else np.nan))
    return pd.DataFrame(rows)


def raw_cube(meta_row):
    arrs = [np.load(str(meta_row[f"file_path_{c}"])) for c in CONDS]
    nf, nt = arrs[0].shape
    return np.concatenate(arrs, axis=1), nf, nt


def draw_raw(ax, A, nf, title, col):
    x = np.linspace(0, len(CONDS), A.shape[1])
    f = np.linspace(0, FMAX, nf)
    im = ax.pcolormesh(x, f, A, cmap=P2.ERSP_CMAP, vmin=-VLIM, vmax=VLIM, shading="auto", rasterized=True)
    for b in range(len(CONDS)):
        ax.axvline(b + 0.5, color=INK, lw=0.8, ls=(0, (4, 3)))
        if b:
            ax.axvline(b, color=INK, lw=1.2)
    ax.set_xlim(0, len(CONDS)); ax.set_ylim(0, FMAX)
    ax.set_xticks([b + 0.5 for b in range(len(CONDS))])
    ax.set_xticklabels(CONDS, fontsize=7.5, color=MUTED)
    ax.set_yticks([0, 100, 200, 300, 400])
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    ax.set_ylabel("Hz", fontsize=7.5)
    ax.set_title(title, fontsize=8.6, loc="left", color=col, pad=4)
    for s_ in ax.spines.values():
        s_.set_color(GREY)
    return im


def page(d, fset, method, cand, regions, top):
    K = d["k"]
    C = P2.cube(d["X"], d)
    means = np.stack([C[d["lab"] == j].mean(0) for j in range(K)])
    order = [int(j) for j in P2.block_order(means, d)[0]] if (fset, method) == ("concat_bands15", "cnmf") else list(range(K))
    vlim = float(np.percentile(np.abs(means), 99.0))
    ex_vlim = float(np.percentile(np.abs(C[cand[cand["rank"] == 1].idx.to_numpy()]), 99.0))
    fig_h = 1.3 + 2.55 * K
    fig = plt.figure(figsize=(17.0, fig_h), dpi=150)
    gs = GridSpec(K, 3, figure=fig, width_ratios=[1.0, 1.0, 2.3], hspace=0.95, wspace=0.22,
                  left=0.045, right=0.985, top=1.0 - 0.75 / fig_h, bottom=0.55 / fig_h)
    fig.suptitle(f"Exemplars   ·   {P2.METHOD_LABEL[method]}   ·   {P2.FS_LABEL[fset]}   ·   K = {K}   ·   "
                 f"the member closest to its cluster's centroid (cosine, unit-norm rows)",
                 x=0.045, y=1.0 - 0.22 / fig_h, ha="left", fontsize=12, color=INK)
    meta = d["meta"]
    dd = dict(conds=d["conds"], bands=d["bands"], nt=d["nt"], fset=fset)
    im_raw = im_mean = None
    records = []
    for r_, j in enumerate(order):
        col = P2.cluster_col(j, K)
        c1 = cand[(cand.cluster == j) & (cand["rank"] == 1)].iloc[0]
        i = int(c1.idx)
        row = meta.iloc[i]
        key = f"{row.patient_id}|{P2.norm(row.electrode)}"
        region = regions.get(key, "")
        nm = P2.type_name(fset, K, j) if method == "cnmf" else ""   # the names belong to the convex NMF fit's ids
        n_j = int((d["lab"] == j).sum())
        npat = int(pd.Series(d["patient"][d["lab"] == j]).nunique())
        # left: the cluster mean
        axm = fig.add_subplot(gs[r_, 0])
        im_mean = P2.draw_mean(axm, means[j], dd, vlim=vlim, col=col, ylim=(-vlim, vlim), show_axes=False, cond_names=False) or im_mean
        axm.set_title(f"c{j}" + (f"  {nm}" if nm else "") + chr(10) + f"cluster mean   ·   n {n_j}   ·   {npat} patients",
                      fontsize=8.6, loc="left", color=col, pad=4)
        if len(d["bands"]) > 1:
            step = 3 if len(d["bands"]) > 6 else 1
            axm.set_yticks(range(0, len(d["bands"]), step))
            axm.set_yticklabels([b.replace("Hz", "") for b in d["bands"][::step]], fontsize=6.4)
            axm.tick_params(axis="y", length=2, colors=MUTED)
        # middle: the exemplar in the feature space
        axe = fig.add_subplot(gs[r_, 1])
        P2.draw_mean(axe, C[i], dd, vlim=ex_vlim, col=col, ylim=(-ex_vlim, ex_vlim), show_axes=False, cond_names=False)
        load_txt = f"   ·   loading {c1.loading:.2f}" if not d["hard"] else ""
        axe.set_title(f"{row.patient_id}  ·  {row.electrode}" + chr(10) + f"in the feature space   ·   cosine {c1.cosine:.2f}{load_txt}",
                      fontsize=8.6, loc="left", color=col, pad=4)
        axe.set_yticks([])
        # right: the raw ERSP
        axr = fig.add_subplot(gs[r_, 2])
        A, nf, nt = raw_cube(row)
        im_raw = draw_raw(axr, A, nf, f"{row.patient_id}  ·  {row.electrode}" + (f"  ·  {region}" if region else "")
                          + chr(10) + "raw ERSP", col)
        records.append(dict(feature_set=fset, method=method, k=K, cluster=j, type_name=nm, n=n_j, n_patients=npat,
                            patient=row.patient_id, electrode=row.electrode, region=region,
                            cosine=round(float(c1.cosine), 4), loading=(round(float(c1.loading), 4) if not d["hard"] else np.nan),
                            file_audio=row.file_path_audio, file_picture=row.file_path_picture, file_reading=row.file_path_reading))
    # colour bars
    if im_raw is not None:
        cax = fig.add_axes([0.905, 0.12 / fig_h, 0.08, 0.10 / fig_h])
        cb = fig.colorbar(im_raw, cax=cax, orientation="horizontal", ticks=[-VLIM, 0, VLIM])
        cb.set_label("raw ERSP, dB re baseline", fontsize=7, color=MUTED, labelpad=2)
        cb.ax.tick_params(labelsize=6.5, colors=MUTED, length=2)
        cb.outline.set_visible(False)
    if im_mean is not None:
        cax2 = fig.add_axes([0.78, 0.12 / fig_h, 0.08, 0.10 / fig_h])
        cb2 = fig.colorbar(im_mean, cax=cax2, orientation="horizontal", ticks=[-round(vlim, 1), 0, round(vlim, 1)])
        cb2.set_label("cluster mean, dB", fontsize=7, color=MUTED, labelpad=2)
        cb2.ax.tick_params(labelsize=6.5, colors=MUTED, length=2)
        cb2.outline.set_visible(False)
    fig.text(0.045, 0.10 / fig_h,
             f"run {d['run'].name}   ·   {len(d['X'])} electrodes, {d['n_patients']} patients   ·   raw ERSP = the 03_ERSP "
             f"files named in the run's labels.csv, 0-{FMAX:.0f} Hz x time-normalised bins per condition, GO cue dashed at "
             f"mid-block   ·   drawn {time.strftime('%Y-%m-%d %H:%M')}",
             fontsize=6.5, color=MUTED)
    return fig, records


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("--feature-set", default=None, help="one feature set (default: all four)")
    ap.add_argument("--method", default=None, choices=METHODS, help="one algorithm (default: all three)")
    ap.add_argument("--top", type=int, default=3, help="candidates per cluster kept in the table")
    a = ap.parse_args()
    cache = LR.current_cache()
    od = OUT / cache
    od.mkdir(parents=True, exist_ok=True)
    regions = region_table()
    fsets = [a.feature_set] if a.feature_set else P2.FSETS
    methods = [a.method] if a.method else METHODS
    index = []
    t0 = time.time()
    for fset in fsets:
        for method in methods:
            d = P2.load_run(fset, K_BY[fset], method)
            d["fset"] = fset
            cand = candidates(d, a.top)
            fig, records = page(d, fset, method, cand, regions, a.top)
            stem = f"exemplars_{fset}_{method}_K{d['k']}{P2.run_tag(d)}"
            P2.save_png(fig, od / f"{stem}.png", facecolor="white", bbox_inches="tight")
            plt.close(fig)
            meta = d["meta"]
            tab = cand.assign(patient=meta.patient_id.to_numpy()[cand.idx], electrode=meta.electrode.to_numpy()[cand.idx],
                              region=[regions.get(f"{p}|{P2.norm(e)}", "") for p, e in
                                      zip(meta.patient_id.to_numpy()[cand.idx], meta.electrode.to_numpy()[cand.idx])],
                              file_audio=meta.file_path_audio.to_numpy()[cand.idx],
                              file_picture=meta.file_path_picture.to_numpy()[cand.idx],
                              file_reading=meta.file_path_reading.to_numpy()[cand.idx]).drop(columns=["idx"])
            P2.save_text(tab.to_csv(index=False), od / f"{stem}.csv")
            index.extend(records)
            print(f"  {FS_SHORT[fset]:9s} x {P2.METHOD_LABEL[method]:10s} K {d['k']:2d}  -> {stem}.png   ({time.time()-t0:.0f} s)")
    P2.save_text(pd.DataFrame(index).to_csv(index=False), od / "exemplars_index.csv")
    print(f"index -> {od.relative_to(ROOT) / 'exemplars_index.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
