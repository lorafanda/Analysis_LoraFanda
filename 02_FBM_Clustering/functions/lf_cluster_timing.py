"""
lf_cluster_timing.py — response-timing analysis for CONCATENATED clusters.

Stage-04 pooling asks "when does each a-priori functional ROLE come on?" and answers
it with a mean-HGA time course per role plus an onset ladder. This module asks the
same question of the DATA-DRIVEN clusters instead:

    when does each cluster's high-gamma response come on, and in what order?

Roles are hypothesis-driven and fixed, so their ordering is a test of prior theory.
Clusters are discovered, so their ordering is a description of what the data actually
separates — and because the concatenated sample is one electrode spanning
[audio | picture | reading], a cluster's three onsets are directly comparable to each
other and to the pooling roles.

The onset definition is deliberately IDENTICAL to lf_pool.plot_role_hga_timeseries:

    onset = first bin whose |mean HGA| crosses `onset_thr_db`, expressed as % of the
            warped trial (50% = GO cue). No crossing -> 100.0 (i.e. "never").

Keeping it identical is the point: it is what makes the cluster ladder and the role
ladder readable on the same axis.

Everything runs over EVERY K in the sweep, not just the chosen one, so the stability
of the ordering across K is visible — an ordering that survives K=4..20 is a property
of the data; one that only appears at the winning K is a property of that partition.

Typical use (see 214_concat_timingranking.ipynb):

    import functions.lf_cluster_timing as T
    labels_by_k = T.load_labels_by_k(run_dir)
    X_hg        = np.load(run_dir / "X_train.npy")      # (n, n_blocks * n_time)
    tab         = T.sweep_timing(X_hg, labels_by_k, out_dir=run_dir / "timing")
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONDITIONS: tuple = ("audio", "picture", "reading")
DEFAULT_ONSET_THR_DB = 0.5          # same default as lf_pool
GO_PCT = 50.0                       # time-warped GO cue


# ============================================================
# Loading
# ============================================================
def load_labels_by_k(run_dir) -> Dict[int, np.ndarray]:
    """{K: labels} from a run's cluster_labels_by_k.csv (written by any K-sweep run)."""
    p = Path(run_dir) / "cluster_labels_by_k.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — that run was fitted at a single K, so there is no sweep "
            f"to walk. Re-fit with params={{'k_range': [...]}} to get one.")
    df = pd.read_csv(p)
    return {int(c.split("_")[1]): df[c].to_numpy()
            for c in df.columns if c.startswith("k_")}


def block_view(X_hg: np.ndarray, n_blocks: int = 3) -> np.ndarray:
    """(n, n_blocks*n_time) -> (n, n_blocks, n_time).

    concat_hg is already the HG line (70-150 Hz mean) stitched across conditions, so
    there is no frequency axis left to collapse — only the time axis to un-stitch.
    """
    X_hg = np.asarray(X_hg, dtype=float)
    if X_hg.ndim != 2:
        raise ValueError(f"expected 2-D (n, n_blocks*n_time), got {X_hg.shape}")
    n, total = X_hg.shape
    if total % n_blocks:
        raise ValueError(f"{total} columns is not divisible by n_blocks={n_blocks}")
    return X_hg.reshape(n, n_blocks, total // n_blocks)


# ============================================================
# Timing
# ============================================================
def cluster_timing(X_hg: np.ndarray, labels: np.ndarray, *,
                   n_blocks: int = 3, conditions: Sequence[str] = CONDITIONS,
                   onset_thr_db: float = DEFAULT_ONSET_THR_DB) -> pd.DataFrame:
    """One row per (cluster, condition): onset, peak, amplitude and size.

    Columns
        cluster, condition, n, onset_pct, peak_pct, peak_db, mean_db, auc_post_go
    `onset_pct` is 100.0 when the cluster never crosses the threshold in that
    condition — that is a real result ("silent here"), not a missing value, so it is
    kept as a number and simply reads as last in the ladder.
    """
    Xb = block_view(X_hg, n_blocks)
    nt = Xb.shape[2]
    t_pct = np.linspace(0, 100, nt, endpoint=False) + 50.0 / nt
    post = t_pct >= GO_PCT

    rows = []
    for c in sorted(np.unique(labels)):
        m = labels == c
        n_c = int(m.sum())
        for bi, cond in enumerate(list(conditions)[:n_blocks]):
            gm = Xb[m, bi, :].mean(axis=0)
            hits = np.where(np.abs(gm) > onset_thr_db)[0]
            pk = int(np.argmax(np.abs(gm)))
            rows.append(dict(
                cluster=int(c), condition=cond, n=n_c,
                onset_pct=float(t_pct[hits[0]]) if len(hits) else 100.0,
                crosses=bool(len(hits)),
                peak_pct=float(t_pct[pk]), peak_db=float(gm[pk]),
                mean_db=float(gm.mean()), auc_post_go=float(gm[post].mean()),
            ))
    return pd.DataFrame(rows)


def sweep_timing(X_hg: np.ndarray, labels_by_k: Dict[int, np.ndarray], *,
                 n_blocks: int = 3, conditions: Sequence[str] = CONDITIONS,
                 onset_thr_db: float = DEFAULT_ONSET_THR_DB,
                 out_dir=None, make_figures: bool = True,
                 dpi: int = 150) -> pd.DataFrame:
    """cluster_timing() for every K in the sweep, stacked into one table.

    Writes per-K figures (time course + onset ladder) plus a cross-K overview when
    `out_dir` is given.
    """
    frames = []
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    for k in sorted(labels_by_k):
        t = cluster_timing(X_hg, labels_by_k[k], n_blocks=n_blocks,
                           conditions=conditions, onset_thr_db=onset_thr_db)
        t.insert(0, "k", k)
        frames.append(t)
        if out and make_figures:
            plot_cluster_hga_timeseries(
                X_hg, labels_by_k[k], n_blocks=n_blocks, conditions=conditions,
                onset_thr_db=onset_thr_db, title=f"K={k}",
                out_png=out / f"timing_k{k:02d}_hga.png", dpi=dpi)
            plot_onset_ladder(t, title=f"Onset ordering · K={k}",
                              out_png=out / f"timing_k{k:02d}_onset.png", dpi=dpi)
    tab = pd.concat(frames, ignore_index=True)
    if out:
        tab.to_csv(out / "cluster_timing_by_k.csv", index=False)
        if make_figures:
            plot_onset_across_k(tab, out_png=out / "onset_across_k.png", dpi=dpi)
    return tab


# ============================================================
# Figures
# ============================================================
def _cluster_colors(clusters) -> Dict[int, tuple]:
    cmap = plt.get_cmap("turbo")
    n = max(len(clusters), 1)
    return {int(c): cmap(0.06 + 0.88 * i / max(n - 1, 1)) for i, c in enumerate(clusters)}


def plot_cluster_hga_timeseries(X_hg: np.ndarray, labels: np.ndarray, *,
                                n_blocks: int = 3, conditions: Sequence[str] = CONDITIONS,
                                onset_thr_db: float = DEFAULT_ONSET_THR_DB,
                                ci_mult: float = 1.0, title: str = "",
                                out_png=None, dpi: int = 150):
    """Mean HGA per cluster, one panel per condition, +/- SEM. Cluster analogue of the
    pooling role time-course figure."""
    Xb = block_view(X_hg, n_blocks)
    nt = Xb.shape[2]
    t_pct = np.linspace(0, 100, nt, endpoint=False) + 50.0 / nt
    clusters = sorted(int(c) for c in np.unique(labels))
    colors = _cluster_colors(clusters)
    conds = list(conditions)[:n_blocks]

    fig, axes = plt.subplots(1, n_blocks, figsize=(5.3 * n_blocks, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for bi, (cond, ax) in enumerate(zip(conds, axes)):
        ax.axhline(0, color="0.75", lw=0.8, zorder=0)
        ax.axhline(onset_thr_db, color="#c0392b", lw=0.7, ls=":", zorder=0)
        ax.axvline(GO_PCT, color="0.4", lw=1.0, ls="--", zorder=0)
        ax.text(GO_PCT + 1.5, 0, "GO", fontsize=7, color="0.4", va="bottom")
        for c in clusters:
            m = labels == c
            mat = Xb[m, bi, :]
            mean = mat.mean(0)
            sem = mat.std(0) / np.sqrt(max(m.sum(), 1)) * ci_mult
            ax.plot(t_pct, mean, color=colors[c], lw=1.5, zorder=2,
                    label=f"c{c} (n={int(m.sum())})" if bi == 0 else None)
            ax.fill_between(t_pct, mean - sem, mean + sem, color=colors[c],
                            alpha=0.13, zorder=1, linewidth=0)
        ax.set_title(cond.capitalize(), fontsize=11)
        ax.set_xlabel("Trial time (%)")
        ax.set_xlim(0, 100)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if bi == 0:
            ax.set_ylabel("HGA (dB)")
            ax.legend(fontsize=7, loc="upper left", ncol=1, framealpha=0.85)
    fig.suptitle(f"Mean HGA per cluster (±1 SEM){'  —  ' + title if title else ''}",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_onset_ladder(timing: pd.DataFrame, *, title: str = "",
                      out_png=None, dpi: int = 150):
    """Onset ladder: clusters on y ordered by mean onset, one panel per condition.

    Ordered by the ACROSS-CONDITION mean so the y-order is the same in all three
    panels — otherwise each panel re-sorts and the comparison the figure exists for
    becomes impossible to make.
    """
    conds = list(dict.fromkeys(timing["condition"]))
    order = (timing.groupby("cluster")["onset_pct"].mean()
             .sort_values().index.tolist())
    colors = _cluster_colors(order)
    pos = {c: i for i, c in enumerate(order)}

    fig, axes = plt.subplots(1, len(conds), figsize=(4.4 * len(conds),
                                                     max(2.6, 0.34 * len(order) + 1.5)),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cond in zip(axes, conds):
        sub = timing[timing["condition"] == cond]
        ax.axvline(GO_PCT, color="0.4", lw=1.0, ls="--", zorder=0)
        for _, r in sub.iterrows():
            y = pos[int(r["cluster"])]
            never = not bool(r.get("crosses", True))
            ax.plot([0, r["onset_pct"]], [y, y], color=colors[int(r["cluster"])],
                    lw=1.2, alpha=0.35, zorder=1)
            ax.scatter(r["onset_pct"], y, s=64,
                       color="white" if never else colors[int(r["cluster"])],
                       edgecolor=colors[int(r["cluster"])], linewidth=1.4,
                       zorder=3, marker="o" if not never else "X")
            if not never:
                ax.text(r["onset_pct"] + 1.6, y, f"{r['onset_pct']:.0f}",
                        va="center", fontsize=7.5, color="#333")
        ax.set_title(cond.capitalize(), fontsize=10.5)
        ax.set_xlim(0, 108)
        ax.set_xlabel("onset (% of trial;  50 = GO)")
        ax.grid(axis="x", alpha=0.25)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"c{c}  (n={int(timing[timing.cluster == c]['n'].iloc[0])})"
                             for c in order], fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle((title or "Onset ordering") +
                 "      ○ = threshold crossed   ✕ = never crosses (plotted at 100)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_onset_across_k(tab: pd.DataFrame, *, out_png=None, dpi: int = 150):
    """Every cluster's onset at every K, one panel per condition.

    The question this answers: is the timing structure a stable property of the data,
    or an artefact of the K we happened to pick? A spread that keeps its shape as K
    grows is the former.
    """
    conds = list(dict.fromkeys(tab["condition"]))
    ks = sorted(tab["k"].unique())
    fig, axes = plt.subplots(1, len(conds), figsize=(4.6 * len(conds), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cond in zip(axes, conds):
        sub = tab[tab["condition"] == cond]
        ax.axhline(GO_PCT, color="0.4", lw=1.0, ls="--", zorder=0)
        for k in ks:
            s = sub[sub["k"] == k]
            jitter = (np.random.default_rng(k).uniform(-0.16, 0.16, len(s)))
            ax.scatter(np.full(len(s), k) + jitter, s["onset_pct"],
                       s=np.clip(s["n"] / 6.0, 8, 90), alpha=0.55,
                       color="#b85c6e", edgecolor="none", zorder=2)
        med = sub.groupby("k")["onset_pct"].median()
        ax.plot(med.index, med.values, color="#22262b", lw=1.4, zorder=3, label="median")
        ax.set_title(cond.capitalize(), fontsize=10.5)
        ax.set_xlabel("K")
        ax.set_xticks(ks[::max(1, len(ks) // 10)])
        ax.grid(alpha=0.25)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0].set_ylabel("onset (% of trial;  50 = GO)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Cluster onset across the whole K sweep  (dot size = cluster n)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def onset_matrix(tab: pd.DataFrame, k: int) -> pd.DataFrame:
    """Wide cluster × condition onset table for one K — the sortable version."""
    s = tab[tab["k"] == k]
    w = s.pivot(index="cluster", columns="condition", values="onset_pct")
    w["mean"] = w.mean(axis=1)
    w["n"] = s.groupby("cluster")["n"].first()
    return w.sort_values("mean")
