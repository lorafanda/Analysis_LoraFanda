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

    Three onsets are reported, because "when did this come on" has two answers on a
    warped trial split by the GO cue at 50%:

        onset_pct        first crossing ANYWHERE — the pooling-identical measure.
                         100.0 when it never crosses (kept as a number so it sorts last).
        onset_stim_pct   first crossing in the STIMULUS window, 0-50% (before GO).
                         NaN when the cluster is silent there.
        onset_resp_pct   first crossing in the RESPONSE window, 50-100% (GO onward).
                         NaN when the cluster is silent there.

    The two windowed onsets are NaN rather than 100.0 on purpose: a cluster with no
    stimulus-locked response has no stimulus onset, and plotting it at 100 would
    invent a late stimulus response that does not exist. Callers skip the NaNs.

    A cluster can legitimately have both (early sensory response, then a separate
    production response), one, or neither.

    Columns
        cluster, condition, n, onset_pct, crosses, onset_stim_pct, onset_resp_pct,
        peak_pct, peak_db, peak_stim_pct, peak_stim_db, peak_resp_pct, peak_resp_db,
        mean_db, auc_post_go
    """
    Xb = block_view(X_hg, n_blocks)
    nt = Xb.shape[2]
    t_pct = np.linspace(0, 100, nt, endpoint=False) + 50.0 / nt
    stim_w = t_pct < GO_PCT
    resp_w = t_pct >= GO_PCT

    rows = []
    for c in sorted(np.unique(labels)):
        m = labels == c
        n_c = int(m.sum())
        for bi, cond in enumerate(list(conditions)[:n_blocks]):
            gm = Xb[m, bi, :].mean(axis=0)
            above = np.abs(gm) > onset_thr_db          # crossing mask
            hits = np.where(above)[0]

            def _first(win):
                h = np.where(above & win)[0]
                return float(t_pct[h[0]]) if len(h) else float("nan")

            def _peak(win):
                idx = np.where(win)[0]
                j = idx[int(np.argmax(np.abs(gm[idx])))]
                return float(t_pct[j]), float(gm[j])

            pk = int(np.argmax(np.abs(gm)))
            ps_t, ps_db = _peak(stim_w)
            pr_t, pr_db = _peak(resp_w)
            # A response onset landing on the very first post-GO bin usually means the
            # signal was ALREADY above threshold at the cue and simply continued — not
            # a new response. Flag it so ~50% is not read as "fires exactly at GO".
            first_resp_bin = int(np.argmax(resp_w))
            resp_carryover = bool(above[first_resp_bin] and above[first_resp_bin - 1]) \
                if first_resp_bin > 0 else False
            rows.append(dict(
                cluster=int(c), condition=cond, n=n_c,
                onset_pct=float(t_pct[hits[0]]) if len(hits) else 100.0,
                crosses=bool(len(hits)),
                onset_stim_pct=_first(stim_w),
                onset_resp_pct=_first(resp_w),
                resp_carryover=resp_carryover,
                peak_pct=float(t_pct[pk]), peak_db=float(gm[pk]),
                peak_stim_pct=ps_t, peak_stim_db=ps_db,
                peak_resp_pct=pr_t, peak_resp_db=pr_db,
                mean_db=float(gm.mean()), auc_post_go=float(gm[resp_w].mean()),
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


def plot_onset_ladder(timing: pd.DataFrame, *, title: str = "", sort_by: str = "stim",
                      out_png=None, dpi: int = 150):
    """Onset ladder with SEPARATE stimulus and response onsets.

    Each cluster row can carry two marks:
        filled circle    stimulus-window onset  (0-50%, before GO)
        filled triangle  response-window onset  (50-100%, GO onward)
    A window with no threshold crossing is simply NOT DRAWN — a cluster that is silent
    before GO has no stimulus onset, and inventing a marker for it would read as a very
    late stimulus response.

    Rows are ordered by the across-condition mean of `sort_by` ("stim", "resp" or
    "any"), and that one order is used in every panel — if each panel re-sorted itself
    the comparison the figure exists for could not be made. Clusters with no onset in
    the sorting window fall to the bottom.
    """
    col = {"stim": "onset_stim_pct", "resp": "onset_resp_pct"}.get(sort_by, "onset_pct")
    conds = list(dict.fromkeys(timing["condition"]))
    key = timing.groupby("cluster")[col].mean()
    order = key.sort_values(na_position="last").index.tolist()
    colors = _cluster_colors(order)
    pos = {c: i for i, c in enumerate(order)}

    fig, axes = plt.subplots(1, len(conds), figsize=(4.6 * len(conds),
                                                     max(2.8, 0.36 * len(order) + 1.7)),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cond in zip(axes, conds):
        sub = timing[timing["condition"] == cond]
        ax.axvspan(0, GO_PCT, color="#f2f4f7", zorder=0)
        ax.axvline(GO_PCT, color="0.35", lw=1.1, ls="--", zorder=1)
        for _, r in sub.iterrows():
            c = int(r["cluster"]); y = pos[c]; col_c = colors[c]
            s_on, r_on = r.get("onset_stim_pct"), r.get("onset_resp_pct")
            have = [v for v in (s_on, r_on) if pd.notna(v)]
            if len(have) == 2:                      # link the pair so it reads as one row
                ax.plot(have, [y, y], color=col_c, lw=1.1, alpha=0.45, zorder=2)
            carry = bool(r.get("resp_carryover", False))
            for v, mk, hollow in ((s_on, "o", False), (r_on, "^", carry)):
                if pd.isna(v):
                    continue
                # hollow triangle = the signal was already above threshold at GO, so
                # this is carry-over, not a new response-locked onset
                ax.scatter(v, y, s=70, marker=mk,
                           color="white" if hollow else col_c,
                           edgecolor=col_c if hollow else "white", linewidth=1.3 if hollow else 0.9,
                           zorder=4)
                ax.text(v + 1.5, y - 0.02, f"{v:.0f}", va="center", fontsize=7.2,
                        color="#333", zorder=5)
            if not have:                            # silent in both windows
                ax.text(2, y, "silent", va="center", fontsize=7.2,
                        style="italic", color="#aab2ba", zorder=3)
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
                 "        ● stimulus window (0-50%)    ▲ response window (50-100%)"
                 "    hollow = already active at GO (carry-over)",
                 fontsize=10, y=1.02)
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
    SERIES = [("onset_stim_pct", "#2f6fb2", "o", "stimulus (0-50%)"),
              ("onset_resp_pct", "#b85c6e", "^", "response (50-100%)")]
    fig, axes = plt.subplots(1, len(conds), figsize=(4.9 * len(conds), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cond in zip(axes, conds):
        sub = tab[tab["condition"] == cond]
        ax.axhspan(0, GO_PCT, color="#f2f4f7", zorder=0)
        ax.axhline(GO_PCT, color="0.35", lw=1.1, ls="--", zorder=1)
        for col, colr, mk, lab in SERIES:
            for k in ks:
                s = sub[(sub["k"] == k) & sub[col].notna()]
                if not len(s):
                    continue
                jitter = np.random.default_rng(k).uniform(-0.16, 0.16, len(s))
                ax.scatter(np.full(len(s), k) + jitter, s[col],
                           s=np.clip(s["n"] / 6.0, 8, 90), alpha=0.5, marker=mk,
                           color=colr, edgecolor="none", zorder=2)
            med = sub.groupby("k")[col].median()
            ax.plot(med.index, med.values, color=colr, lw=1.5, zorder=3, label=lab)
        ax.set_title(cond.capitalize(), fontsize=10.5)
        ax.set_xlabel("K")
        ax.set_xticks(ks[::max(1, len(ks) // 10)])
        ax.grid(alpha=0.25)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0].set_ylabel("onset (% of trial;  50 = GO)")
    axes[0].legend(fontsize=8, loc="center left")
    fig.suptitle("Cluster onset across the whole K sweep — stimulus vs response window  "
                 "(dot size = cluster n; a window with no crossing is not plotted)",
                 fontsize=10.5, y=1.02)
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
