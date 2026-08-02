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
            order = onset_order(t)
            plot_cluster_hga_timeseries(
                X_hg, labels_by_k[k], n_blocks=n_blocks, conditions=conditions,
                onset_thr_db=onset_thr_db, title=f"K={k}", order=order,
                out_png=out / f"timing_k{k:02d}_hga.png", dpi=dpi)
            plot_onset_ladder(t, title=f"Onset ordering · K={k}", order=order,
                              out_png=out / f"timing_k{k:02d}_onset.png", dpi=dpi)
            plot_cluster_timing_panel(
                X_hg, labels_by_k[k], t, n_blocks=n_blocks, conditions=conditions,
                onset_thr_db=onset_thr_db, title=f"K={k}",
                out_png=out / f"timing_k{k:02d}_panel.png", dpi=dpi)
    tab = pd.concat(frames, ignore_index=True)
    if out:
        tab.to_csv(out / "cluster_timing_by_k.csv", index=False)
        if make_figures:
            plot_onset_across_k(tab, labels_by_k=labels_by_k,
                                out_png=out / "onset_across_k.png", dpi=dpi)
    return tab


# ============================================================
# Figures
# ============================================================
def _cluster_colors(clusters) -> Dict[int, tuple]:
    """Colour by POSITION in the given sequence, so the caller controls the mapping.

    Pass the onset-sorted order and the colours become a timing gradient: earliest
    cluster at one end of turbo, latest at the other. Pass the same order to every
    figure of a run and a cluster keeps one colour throughout — which is the only way
    the time-course panel and the ladder can be read together.
    """
    cmap = plt.get_cmap("turbo")
    n = max(len(clusters), 1)
    return {int(c): cmap(0.06 + 0.88 * i / max(n - 1, 1)) for i, c in enumerate(clusters)}


def onset_order(timing: pd.DataFrame, sort_by: str = "stim") -> list:
    """Cluster ids ordered by across-condition mean onset. The canonical ordering —
    use it for both the colour map and the ladder rows so the two never disagree."""
    col = {"stim": "onset_stim_pct", "resp": "onset_resp_pct"}.get(sort_by, "onset_pct")
    return (timing.groupby("cluster")[col].mean()
            .sort_values(na_position="last").index.astype(int).tolist())


def plot_cluster_hga_timeseries(X_hg: np.ndarray, labels: np.ndarray, *,
                                n_blocks: int = 3, conditions: Sequence[str] = CONDITIONS,
                                onset_thr_db: float = DEFAULT_ONSET_THR_DB,
                                ci_mult: float = 1.0, title: str = "",
                                order: Optional[Sequence[int]] = None,
                                width_per_panel: float = 7.2,
                                out_png=None, dpi: int = 150):
    """Mean HGA per cluster, one panel per condition, +/- SEM.

    `order` sets the colour mapping — pass onset_order(timing) so the curves carry the
    same timing-gradient colours as the ladder. Without it, colours fall back to
    cluster-id order and will NOT match the ladder.
    """
    Xb = block_view(X_hg, n_blocks)
    nt = Xb.shape[2]
    t_pct = np.linspace(0, 100, nt, endpoint=False) + 50.0 / nt
    clusters = sorted(int(c) for c in np.unique(labels))
    colors = _cluster_colors([int(c) for c in order] if order is not None else clusters)
    conds = list(conditions)[:n_blocks]

    fig, axes = plt.subplots(1, n_blocks, figsize=(width_per_panel * n_blocks, 4.6),
                             sharey=True, sharex=True)
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
                      order: Optional[Sequence[int]] = None,
                      width_per_panel: float = 7.2, out_png=None, dpi: int = 150):
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
    conds = list(dict.fromkeys(timing["condition"]))
    order = [int(c) for c in order] if order is not None else onset_order(timing, sort_by)
    colors = _cluster_colors(order)
    pos = {c: i for i, c in enumerate(order)}

    fig, axes = plt.subplots(1, len(conds),
                             figsize=(width_per_panel * len(conds),
                                      max(2.8, 0.36 * len(order) + 1.7)),
                             sharey=True, sharex=True)
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


def onset_matrix(tab: pd.DataFrame, k: int) -> pd.DataFrame:
    """Wide cluster × condition onset table for one K — the sortable version."""
    s = tab[tab["k"] == k]
    w = s.pivot(index="cluster", columns="condition", values="onset_pct")
    w["mean"] = w.mean(axis=1)
    w["n"] = s.groupby("cluster")["n"].first()
    return w.sort_values("mean")


def plot_cluster_timing_panel(X_hg: np.ndarray, labels: np.ndarray,
                              timing: pd.DataFrame, *,
                              n_blocks: int = 3, conditions: Sequence[str] = CONDITIONS,
                              onset_thr_db: float = DEFAULT_ONSET_THR_DB,
                              sort_by: str = "stim", ci_mult: float = 1.0,
                              title: str = "", width_per_panel: float = 7.2,
                              out_png=None, dpi: int = 150):
    """Time courses and onset ladder in ONE figure, sharing the x axis per condition.

    The two were separate figures on the same 0-100% axis, which meant reading a
    cluster's onset off the ladder and then hunting for the matching curve above it.
    Stacked and sharex'd, an onset marker sits directly under the curve it came from.

    Colours are the onset gradient: clusters are coloured by their rank in `sort_by`
    order, so early clusters sit at one end of turbo and late ones at the other, and
    the same cluster is the same colour in both rows.
    """
    Xb = block_view(X_hg, n_blocks)
    nt = Xb.shape[2]
    t_pct = np.linspace(0, 100, nt, endpoint=False) + 50.0 / nt
    conds = list(conditions)[:n_blocks]
    order = onset_order(timing, sort_by)
    colors = _cluster_colors(order)
    pos = {c: i for i, c in enumerate(order)}

    h_top = 4.4
    h_bot = max(2.2, 0.32 * len(order) + 1.1)
    # sharey='row': the top row gets ONE dB scale across conditions (so curve heights
    # are comparable, not silently rescaled per panel) and the bottom row shares the
    # cluster positions, which also suppresses the stray 0..K-1 ticks on panels 2-3.
    fig, axes = plt.subplots(
        2, n_blocks, sharex=True, sharey='row',
        figsize=(width_per_panel * n_blocks, h_top + h_bot),
        gridspec_kw={"height_ratios": [h_top, h_bot], "hspace": 0.08})
    axes = np.atleast_2d(axes)

    for bi, cond in enumerate(conds):
        # ---- top: mean HGA per cluster
        ax = axes[0, bi]
        ax.axvspan(0, GO_PCT, color="#f2f4f7", zorder=0)
        ax.axhline(0, color="0.75", lw=0.8, zorder=1)
        ax.axhline(onset_thr_db, color="#c0392b", lw=0.7, ls=":", zorder=1)
        ax.axhline(-onset_thr_db, color="#c0392b", lw=0.7, ls=":", zorder=1)
        ax.axvline(GO_PCT, color="0.35", lw=1.1, ls="--", zorder=1)
        for c in order:
            m = labels == c
            if not m.any():
                continue
            mat = Xb[m, bi, :]
            mean = mat.mean(0)
            sem = mat.std(0) / np.sqrt(max(m.sum(), 1)) * ci_mult
            ax.plot(t_pct, mean, color=colors[c], lw=1.6, zorder=3,
                    label=f"c{c} (n={int(m.sum())})" if bi == 0 else None)
            ax.fill_between(t_pct, mean - sem, mean + sem, color=colors[c],
                            alpha=0.13, zorder=2, linewidth=0)
        ax.set_title(cond.capitalize(), fontsize=11.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if bi == 0:
            ax.set_ylabel("HGA (dB)")
            ax.legend(fontsize=7.5, loc="upper left", ncol=1, framealpha=0.85)
        ax.text(GO_PCT + 1.2, ax.get_ylim()[1] * 0.94, "GO", fontsize=7.5, color="0.35")

        # ---- bottom: the ladder, same x
        axb = axes[1, bi]
        sub = timing[timing["condition"] == cond]
        axb.axvspan(0, GO_PCT, color="#f2f4f7", zorder=0)
        axb.axvline(GO_PCT, color="0.35", lw=1.1, ls="--", zorder=1)
        for _, r in sub.iterrows():
            c = int(r["cluster"])
            if c not in pos:
                continue
            y = pos[c]
            s_on, r_on = r.get("onset_stim_pct"), r.get("onset_resp_pct")
            carry = bool(r.get("resp_carryover", False))
            have = [v for v in (s_on, r_on) if pd.notna(v)]
            if len(have) == 2:
                axb.plot(have, [y, y], color=colors[c], lw=1.1, alpha=0.45, zorder=2)
            for v, mk, hollow in ((s_on, "o", False), (r_on, "^", carry)):
                if pd.isna(v):
                    continue
                axb.scatter(v, y, s=68, marker=mk,
                            color="white" if hollow else colors[c],
                            edgecolor=colors[c] if hollow else "white",
                            linewidth=1.3 if hollow else 0.9, zorder=4)
                axb.text(v + 1.4, y - 0.02, f"{v:.0f}", va="center", fontsize=7,
                         color="#333", zorder=5)
            if not have:
                axb.text(2, y, "silent", va="center", fontsize=7,
                         style="italic", color="#aab2ba", zorder=3)
        axb.set_xlim(0, 104)
        axb.set_xlabel("trial time (%)   —   50 = GO")
        axb.grid(axis="x", alpha=0.25)
        for s in ("top", "right", "left"):
            axb.spines[s].set_visible(False)

    axes[1, 0].set_yticks(range(len(order)))
    axes[1, 0].set_yticklabels(
        [f"c{c}  (n={int(timing[timing.cluster == c]['n'].iloc[0])})" for c in order],
        fontsize=8.5)
    axes[1, 0].invert_yaxis()
    fig.suptitle((title or "Cluster response timing") +
                 "        ● stimulus onset    ▲ response onset    "
                 "hollow ▲ = already active at GO    absent = no crossing",
                 fontsize=11, y=0.995)
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


# ============================================================
# Across-K: identity-tracked, with the size confound made explicit
# ============================================================
def match_clusters_across_k(labels_by_k: Dict[int, np.ndarray]) -> pd.DataFrame:
    """Follow each cluster from K to K+1 by maximum sample overlap (Jaccard).

    Without this there is no such thing as "the ordering across K": a scatter of
    onsets at each K has no identity, so nothing can be said to be stable or unstable.
    Matching gives every cluster a LINEAGE id that persists across the sweep, and the
    onset of a lineage can then actually be tracked.

    Returns columns: k, cluster, lineage, jaccard (overlap with its K-1 parent).
    Greedy and one-directional: at each step every cluster is assigned to the parent it
    overlaps most. Splits therefore share a lineage until one child wins it; that is a
    real limitation, and `jaccard` is what tells you how much to trust the link.
    """
    ks = sorted(labels_by_k)
    rows, prev_line = [], {}
    for i, k in enumerate(ks):
        lab = np.asarray(labels_by_k[k])
        if i == 0:
            for c in sorted(np.unique(lab)):
                prev_line[int(c)] = int(c)
                rows.append(dict(k=k, cluster=int(c), lineage=int(c), jaccard=1.0))
            prev_lab = lab
            continue
        cur_line = {}
        for c in sorted(np.unique(lab)):
            m = lab == c
            best, best_j = None, -1.0
            for pc in np.unique(prev_lab):
                pm = prev_lab == pc
                inter = np.logical_and(m, pm).sum()
                if not inter:
                    continue
                j = inter / np.logical_or(m, pm).sum()
                if j > best_j:
                    best, best_j = int(pc), float(j)
            lin = prev_line.get(best, int(c)) if best is not None else int(c)
            cur_line[int(c)] = lin
            rows.append(dict(k=k, cluster=int(c), lineage=lin,
                             jaccard=round(best_j, 3) if best is not None else np.nan))
        prev_line, prev_lab = cur_line, lab
    return pd.DataFrame(rows)


def plot_onset_across_k(tab: pd.DataFrame, *, labels_by_k: Optional[Dict] = None,
                        out_png=None, dpi: int = 150):
    """Is the timing structure a property of the DATA or of the K we picked?

    Three rows, because the plain scatter version of this figure invited a conclusion
    it could not support:

      1  LINEAGE TRACES. Clusters matched across K by overlap, so each line follows one
         lineage. A cloud of unlinked dots cannot show stability, because stability is
         a statement about identity. Needs `labels_by_k`; falls back to the scatter.
      2  THE SIZE CONFOUND. Onset against cluster n. A small cluster has a noisier mean,
         which crosses a fixed dB threshold sooner, so onset drifts EARLIER as K grows
         and clusters shrink. On the concatenated run this is Spearman r = +0.42
         (p ~ 1e-20) — not a nuisance, a first-order effect. Any apparent "spread grows
         with K" has to be read against it.
      3  COVERAGE. The fraction of clusters with any onset at all in each window. The
         scatter silently omits NaNs, so a K where half the clusters are silent looks
         sparse rather than uninformative.
    """
    conds = list(dict.fromkeys(tab["condition"]))
    ks = sorted(tab["k"].unique())
    SERIES = [("onset_stim_pct", "#2f6fb2", "o", "stimulus (0-50%)"),
              ("onset_resp_pct", "#b85c6e", "^", "response (50-100%)")]

    lin = match_clusters_across_k(labels_by_k) if labels_by_k else None
    tt = tab.merge(lin, on=["k", "cluster"], how="left") if lin is not None else tab.copy()

    fig, axes = plt.subplots(3, len(conds), figsize=(5.4 * len(conds), 11.4),
                             gridspec_kw={"height_ratios": [1.25, 1.0, 0.75],
                                          "hspace": 0.30})
    axes = np.atleast_2d(axes)

    for ci, cond in enumerate(conds):
        sub = tt[tt["condition"] == cond]

        # --- 1. lineage traces
        ax = axes[0, ci]
        ax.axhspan(0, GO_PCT, color="#f2f4f7", zorder=0)
        ax.axhline(GO_PCT, color="0.35", lw=1.1, ls="--", zorder=1)
        if "lineage" in sub.columns and sub["lineage"].notna().any():
            lineages = sorted(sub["lineage"].dropna().unique())
            cmap = plt.get_cmap("turbo")
            for li, lg in enumerate(lineages):
                s = sub[sub["lineage"] == lg]
                col = cmap(0.06 + 0.88 * li / max(len(lineages) - 1, 1))
                # A lineage holds SEVERAL clusters once it has split, so plotting every
                # member gives one zig-zag line per lineage instead of a trajectory.
                # Trace the dominant heir (largest cluster) and show the rest as faint
                # dots, so a split is visible as spread around its own trace.
                heir = s.loc[s.groupby("k")["n"].idxmax()].sort_values("k")
                rest = s.drop(index=heir.index)
                for c_, style in (("onset_stim_pct", "-"), ("onset_resp_pct", "--")):
                    d = heir.dropna(subset=[c_])
                    if len(d) > 1:
                        ax.plot(d["k"], d[c_], style, color=col, lw=1.6, alpha=.9,
                                zorder=3)
                    r_ = rest.dropna(subset=[c_])
                    if len(r_):
                        ax.scatter(r_["k"], r_[c_], s=9, color=col, alpha=.30,
                                   edgecolor="none", zorder=2)
            ax.set_title(f"{cond.capitalize()} — lineage traces "
                         f"({len(lineages)} lineages; line = dominant heir, dots = split-off)",
                         fontsize=9.5)
        else:
            for col, colr, mk, lab_ in SERIES:
                s = sub.dropna(subset=[col])
                ax.scatter(s["k"], s[col], s=18, marker=mk, color=colr, alpha=.5)
            ax.set_title(f"{cond.capitalize()} — (no lineage: pass labels_by_k)",
                         fontsize=10.5)
        ax.set_xlabel("K")
        if ci == 0:
            ax.set_ylabel("onset (%)   solid = stim, dashed = resp")

        # --- 2. the size confound
        ax = axes[1, ci]
        for col, colr, mk, lab_ in SERIES:
            s = sub.dropna(subset=[col])
            ax.scatter(s["n"], s[col], s=14, marker=mk, color=colr, alpha=.45,
                       edgecolor="none", label=lab_)
        s_all = sub.dropna(subset=["onset_stim_pct"])
        if len(s_all) > 8:
            from scipy.stats import spearmanr
            r, pv = spearmanr(s_all["n"], s_all["onset_stim_pct"])
            ax.text(.97, .04, f"stim: rho={r:+.2f}\np={pv:.1e}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8,
                    bbox=dict(fc="white", ec="#ccc", alpha=.85, boxstyle="round,pad=.3"))
        ax.set_xscale("log")
        ax.set_xlabel("cluster n (log)")
        ax.set_title("onset vs cluster size — the confound", fontsize=10)
        if ci == 0:
            ax.set_ylabel("onset (%)")
            ax.legend(fontsize=7.5, loc="upper left")

        # --- 3. coverage
        ax = axes[2, ci]
        cov = sub.groupby("k").agg(
            stim=("onset_stim_pct", lambda v: 100 * v.notna().mean()),
            resp=("onset_resp_pct", lambda v: 100 * v.notna().mean()))
        ax.plot(cov.index, cov["stim"], "-o", ms=3, color="#2f6fb2", label="stimulus")
        ax.plot(cov.index, cov["resp"], "-^", ms=3, color="#b85c6e", label="response")
        ax.set_ylim(0, 105)
        ax.set_xlabel("K")
        ax.set_title("% of clusters with any onset", fontsize=10)
        if ci == 0:
            ax.set_ylabel("coverage (%)")
            ax.legend(fontsize=7.5)

    for ax in axes.ravel():
        ax.grid(alpha=.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Onset across the K sweep — tracked, confound-checked, coverage-aware",
                 fontsize=12, y=0.997)
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig

# ============================================================
# Cross-correlation against a reference cluster
# ============================================================
# Sign convention, verified on synthetic data in both directions:
#
#     r(tau) = Pearson[ ref(t), other(t + tau) ]
#     tau > 0  ->  the OTHER cluster happens LATER  (the reference leads)
#
# WINDOW MATTERS MORE THAN ANY OTHER SETTING. Cross-correlation only means
# something when two signals are the same SHAPE shifted in time. Over the full
# trial these cluster means are not: one has an early sensory transient, another
# only a late production bump, so no shift maps one onto the other and the peak
# lag is whatever maximises a weak correlation. Measured on run 20260802_220720
# K=10, the full-trial lag disagreed with the actual peak-time difference for
# 8 of 8 clusters. Restricted to the RESPONSE window every cluster has a
# production-period bump, the shapes genuinely are shifted copies, and 8 of 8
# agree. Use window="response"; "full" is kept for diagnosis, not for reporting.
#
# MIN_OVERLAP sets the usable lag range, not `max_lag`: at overlap f the widest
# lag is (1-f) * window length. 0.6 was chosen empirically — 0.8 censored half
# the peaks at the search boundary, 0.6 censored none.
DEFAULT_MIN_OVERLAP = 0.6
DEFAULT_SHAPE_GATE = 0.30      # peak r below this -> shapes differ, lag is not a delay


def cross_correlate(a: np.ndarray, b: np.ndarray, *,
                    min_overlap: float = DEFAULT_MIN_OVERLAP):
    """Lag profile of Pearson r between two 1-D signals.

    Returns (lags_in_bins, r). r is NaN where either segment is constant.
    Pearson is recomputed on the overlapping segment at each lag rather than
    zero-padding, so a lag is never rewarded for padding one signal with zeros.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    max_lag = int(len(a) * (1.0 - min_overlap))
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.full(len(lags), np.nan)
    for i, L in enumerate(lags):
        x, y = (a[:len(a) - L], b[L:]) if L >= 0 else (a[-L:], b[:len(b) + L])
        if len(x) < 3 or x.std() == 0 or y.std() == 0:
            continue
        out[i] = np.corrcoef(x, y)[0, 1]
    return lags, out


def _window_slice(nt: int, window: str) -> slice:
    if window == "response":
        return slice(nt // 2, None)          # 50-100%, GO onward
    if window == "stimulus":
        return slice(0, nt // 2)
    if window == "full":
        return slice(None)
    raise ValueError(f"window must be full / response / stimulus, got {window!r}")


def cluster_xcorr(X_hg: np.ndarray, labels: np.ndarray, ref_cluster: int, *,
                  n_blocks: int = 3, conditions: Sequence[str] = CONDITIONS,
                  window: str = "response",
                  min_overlap: float = DEFAULT_MIN_OVERLAP,
                  n_boot: int = 400, n_shuffle: int = 200,
                  shape_gate: float = DEFAULT_SHAPE_GATE,
                  seed: int = 0) -> tuple:
    """Every cluster's lag profile against `ref_cluster`, per condition.

    Returns (table, profiles) where
      table    one row per (condition, cluster): peak_lag_pct, peak_r, bootstrap CI,
               shuffle p, and `interpretable` — the honest gate.
      profiles {(condition, cluster): (lag_pct, r)} for plotting the full curve.

    Three things are reported alongside the lag because the lag alone is not
    trustworthy on its own:
      peak_r        how well the shapes align AT ALL. Below `shape_gate` the two
                    clusters simply have different shapes and the lag is not a delay.
      ci_lo / ci_hi bootstrap over electrodes WITHIN each cluster — the precision of
                    the lag. A wide CI means the lag is not resolvable, whatever its
                    point estimate.
      at_edge       the peak sits at the end of the searched range, so the true peak
                    may lie beyond it. A censored lag, not a measured one.
      p_shuffle     label-permutation null ON THE LAG. Note this is a WEAK test:
                    shuffling makes both means near-identical so the null lag is ~0
                    and almost any real lag beats it. It rules out "lag is zero",
                    nothing more. (The same shuffle applied to peak r is useless —
                    it gives p ~ 1.0 because the shuffled means correlate at ~0.97.)
    """
    rng = np.random.default_rng(seed)
    Xb = block_view(X_hg, n_blocks)
    nt = Xb.shape[2]
    sl = _window_slice(nt, window)
    conds = list(conditions)[:n_blocks]
    clusters = sorted(int(c) for c in np.unique(labels))
    if ref_cluster not in clusters:
        raise ValueError(f"ref_cluster {ref_cluster} not in {clusters}")
    idx = {c: np.where(labels == c)[0] for c in clusters}

    rows, profiles = [], {}
    for bi, cond in enumerate(conds):
        ref_mean = Xb[idx[ref_cluster], bi, sl].mean(0)
        for c in clusters:
            if c == ref_cluster:
                continue
            other = Xb[idx[c], bi, sl].mean(0)
            lags, r = cross_correlate(ref_mean, other, min_overlap=min_overlap)
            profiles[(cond, c)] = (lags / nt * 100.0, r)
            if np.all(np.isnan(r)):
                continue
            j = int(np.nanargmax(r))
            peak_lag, peak_r = int(lags[j]), float(r[j])
            at_edge = abs(peak_lag) >= abs(lags[-1]) - 1

            boot = np.empty(n_boot)
            for k in range(n_boot):
                aa = Xb[rng.choice(idx[ref_cluster], len(idx[ref_cluster]), replace=True), bi, sl].mean(0)
                bb = Xb[rng.choice(idx[c], len(idx[c]), replace=True), bi, sl].mean(0)
                lg, rr = cross_correlate(aa, bb, min_overlap=min_overlap)
                boot[k] = lg[int(np.nanargmax(rr))] if not np.all(np.isnan(rr)) else np.nan
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])

            null = np.empty(n_shuffle)
            for k in range(n_shuffle):
                sh = rng.permutation(labels)
                aa = Xb[sh == ref_cluster, bi, sl].mean(0)
                bb = Xb[sh == c, bi, sl].mean(0)
                lg, rr = cross_correlate(aa, bb, min_overlap=min_overlap)
                null[k] = abs(lg[int(np.nanargmax(rr))]) if not np.all(np.isnan(rr)) else np.nan
            p_sh = float(np.nanmean(null >= abs(peak_lag)))

            ci_w = (hi - lo) / nt * 100.0
            rows.append(dict(
                condition=cond, cluster=c, n=int(len(idx[c])),
                ref=ref_cluster, window=window,
                peak_lag_pct=peak_lag / nt * 100.0, peak_r=peak_r,
                ci_lo_pct=lo / nt * 100.0, ci_hi_pct=hi / nt * 100.0, ci_width_pct=ci_w,
                at_edge=at_edge, p_shuffle=p_sh,
                interpretable=bool(peak_r >= shape_gate and not at_edge and ci_w <= 20.0),
                reason=("shapes differ (r<%.2f)" % shape_gate if peak_r < shape_gate else
                        "lag censored at search edge" if at_edge else
                        "CI too wide" if ci_w > 20.0 else "ok"),
            ))
    return pd.DataFrame(rows), profiles


def plot_xcorr_profiles(table: pd.DataFrame, profiles: dict, *,
                        ref_cluster: int, order: Optional[Sequence[int]] = None,
                        shape_gate: float = DEFAULT_SHAPE_GATE,
                        title: str = "", width_per_panel: float = 7.2,
                        out_png=None, dpi: int = 150):
    """Full r-vs-lag curve per cluster, one panel per condition, peak marked.

    Curves that fail the interpretability gate are drawn dashed and pale — they are
    shown rather than hidden, because "these two shapes do not align at any lag" is
    a result, but they must not read like a measured delay.
    """
    conds = list(dict.fromkeys(table["condition"]))
    clusters = sorted(table["cluster"].unique())
    order = [int(c) for c in order] if order is not None else clusters
    colors = _cluster_colors([c for c in order if c != ref_cluster] or clusters)

    fig, axes = plt.subplots(1, len(conds), figsize=(width_per_panel * len(conds), 5.0),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, cond in zip(axes, conds):
        ax.axhline(0, color="0.75", lw=0.8, zorder=0)
        ax.axvline(0, color="0.35", lw=1.1, ls="--", zorder=1)
        ax.axhspan(-shape_gate, shape_gate, color="#f2f4f7", zorder=0)
        for c in clusters:
            if (cond, c) not in profiles:
                continue
            lag, r = profiles[(cond, c)]
            row = table[(table.condition == cond) & (table.cluster == c)]
            good = bool(row["interpretable"].iloc[0]) if len(row) else False
            col = colors.get(c, "#888")
            ax.plot(lag, r, color=col, lw=1.8 if good else 1.0,
                    ls="-" if good else "--", alpha=1.0 if good else .45, zorder=3 if good else 2,
                    label=f"c{c} (n={int(row['n'].iloc[0])})" if len(row) else f"c{c}")
            if len(row) and good:
                x = float(row["peak_lag_pct"].iloc[0]); y = float(row["peak_r"].iloc[0])
                ax.scatter([x], [y], s=70, color=col, edgecolor="white", lw=1.2, zorder=5)
                ax.annotate(f"{x:+.0f}%", (x, y), textcoords="offset points", xytext=(4, 5),
                            fontsize=8, color=col, fontweight="bold")
        ax.set_title(cond.capitalize(), fontsize=11)
        ax.set_xlabel(f"lag (% of trial)   + = cluster LAGS behind c{ref_cluster}")
        ax.grid(alpha=.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Pearson r")
    axes[0].legend(fontsize=7.5, loc="lower left", ncol=2, framealpha=.9)
    fig.suptitle((title or f"Cross-correlation against c{ref_cluster}") +
                 f"        solid = interpretable   dashed = shapes differ (|r| < {shape_gate}) "
                 f"or lag censored   grey band = below the shape gate",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig
