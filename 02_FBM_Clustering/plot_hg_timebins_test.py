#!/usr/bin/env python3
"""
plot_hg_timebins_test.py - the two pictures of a run_hg_timebins_test run, and a per-electrode table.

    python plot_hg_timebins_test.py outputs/clustering/hg_timebins_test/<run_id> [--set hg60]

WHAT IT ANSWERS
  G1  what changes with the grid, and what does not
      row 1  today's seven components with every set's matched component drawn over them
             (the HG course through audio | picture | reading): the curves that do not move
      row 2  (a) ARI between every pair of partitions, the published fit and the sweep's refit
             included, so the fit-to-fit floor sits next to the grids
             (b) share of electrodes whose label matches the published fit, by margin bin,
             one line per set, with the margin histogram behind: where the flips live
             (c) three reliabilities per set: split-half by refit (the test's number), split-half
             with the components held fixed (each half assigned by non-negative least squares
             to the full fit's components), and the three-way agreement full fit = half 1 = half 2
  G2  which electrodes can be trusted (one set, --set, default hg60)
      row 1  margin against HG response amplitude, reliable electrodes in colour, the rest grey
      row 2  the gate: share reliable among the electrodes with margin >= m, and how many are
             retained, for every set
      row 3  fsaverage: the reliable electrodes coloured by cluster, the unreliable in grey
  electrodes.tsv   one row per electrode: keys, published label, every set's label matched to the
                   published ids, margin, the two half labels, the reliable flag, HG amplitude, xyz
  reliability.tsv  one row per set
  <run>/reliability.csv   beside the compared run's labels.csv (--publish-set, default hg300 = the run's
                   own features): per electrode reliable / margin / half labels, for the cluster
                   visualizer (make_coverage_bundle.py puts it into contacts.json, the chips show it)

"Reliable" = the electrode gets the same component from the full fit and from each trial half,
the halves assigned with the FULL fit's components held fixed. That isolates trial noise from the
fit-to-fit jitter of the argmax (which the refit split-half of the test compounds).
Nothing is refitted here; the components come from the test run, the halves from the counters cache.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from scipy.optimize import nnls                       # noqa: E402
from sklearn.metrics import adjusted_rand_score as ARI  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from functions import lf_concat as CC                 # noqa: E402
from functions import lf_counters as LCn              # noqa: E402
from functions import lf_decompose as D               # noqa: E402
from run_hg_timebins_test import bins_of, feature_tensor, flat, hg_part, hg_tensor, N_COND, N_TIME   # noqa: E402

SET_COL = {"hg300": "#111111", "hg150": "#6a51a3", "hg100": "#2171b5", "hg60": "#238b45", "hg30": "#d94801", "lowhg30": "#c51b7d",
           "lowhg60": "#e377c2", "lowhg100": "#f7b6d2", "lowhg150": "#fbc9df"}
THRESH = np.round(np.arange(0.0, 0.41, 0.02), 2)


def key_of(df: pd.DataFrame) -> list:
    return list(df["patient_id"].astype(str) + "|" + df["contact_norm"].astype(str))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--set", default="hg60", help="the set of G2's electrode view")
    ap.add_argument("--publish-set", default="hg300", help="whose reliability goes beside the run's labels.csv for the cluster visualizer (hg300 = the run's own features); 'none' to skip")
    a = ap.parse_args()
    out = Path(a.run_dir)
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    sets = [s for s in man["sets"] if (out / f"components_{s}.npy").exists()]
    K, prep = int(man["K"]), man.get("preprocessing", "unit_norm")
    lab_t = pd.read_csv(out / "labels.csv")
    summ = pd.read_csv(out / "summary.tsv", sep="\t").set_index("set")
    run = Path(man["compared_run"])
    pub = pd.read_csv(run / "labels.csv")
    by_k = pd.read_csv(run / "cluster_labels_by_k.csv")
    pub_col = [c for c in pub.columns if c.startswith("cluster_")][0]
    today_C = np.load(run / "components.npy")                         # (K_run, 900)
    if today_C.shape[0] != K:
        print(f"the run's K ({today_C.shape[0]}) is not the test's K ({K}): G1 row 1 and the published comparison need equal K")
        return 1

    # ---- the cohort in the test's order, its halves, its HG amplitude -------------------------------
    df, X, cache = LCn.load_cohort(verbose=False)
    pos = {k: i for i, k in enumerate(key_of(df))}
    keys_t = key_of(lab_t)
    order = np.array([pos[k] for k in keys_t])
    df, X = df.iloc[order].reset_index(drop=True), X[order]
    assert key_of(pub) == keys_t[:len(pub)] or set(keys_t) <= set(key_of(pub)), "the test's electrodes are not the run's"
    pub = pub.set_index(pd.Index(key_of(pub))).loc[keys_t].reset_index(drop=True)
    by_k = by_k.iloc[[{k: i for i, k in enumerate(key_of(pd.read_csv(run / 'labels.csv')))}[k] for k in keys_t]].reset_index(drop=True)
    n = len(df)
    patient = df["patient_id"].astype(str).to_numpy()
    X1, X2 = LCn.load_halves(df, cache["name"], verbose=False)
    ok = ~(np.isnan(X1).any((1, 2)) | np.isnan(X2).any((1, 2)))
    hg900 = CC.concat_hg_features(X).astype(np.float64)
    hg1 = CC.concat_hg_features(np.nan_to_num(X1)).astype(np.float64)
    hg2 = CC.concat_hg_features(np.nan_to_num(X2)).astype(np.float64)
    hg_rows = [r for r in range(103) if 70 <= LCn.FREQ[r] < 150 and r not in LCn.HARM_ROWS]
    amp = np.max([X[:, hg_rows, b * N_TIME + 150:(b + 1) * N_TIME].mean((1, 2)) for b in range(N_COND)], axis=0)   # dB, response half, best condition
    xyz = LCn.coords_for(df)

    # ---- labels of every partition, matched onto the published ids -----------------------------------
    today = pub[pub_col].to_numpy(int)
    today_margin = pub["margin"].to_numpy(float) if "margin" in pub else np.full(n, np.nan)
    parts = {"published": today, "sweep refit": by_k[f"k_{K}"].to_numpy(int)}
    matched, margins = {}, {}
    for s in sets:
        raw = lab_t[f"{s}_label"].to_numpy(int)
        matched[s], _ = LCn.match_labels(today, raw)
        margins[s] = lab_t[f"{s}_margin"].to_numpy(float)
        parts[s] = raw
    names = list(parts)
    A = np.array([[ARI(parts[i], parts[j]) for j in names] for i in names])

    # ---- the halves assigned to the fixed components ---------------------------------------------------
    rel_rows, reliable, half_lab = [], {}, {}
    for s in sets:
        C = np.load(out / f"components_{s}.npy").astype(np.float64)           # (K, F), in the set's own ids
        raw = parts[s]
        bm, _ = LCn.match_labels(today, raw)
        remap = {}
        for j in np.unique(raw):
            remap[int(j)] = int(np.bincount(bm[raw == j]).argmax())
        B = C.T                                                                 # F x K
        labs = []
        for Xh, hgh in ((X1, hg1), (X2, hg2)):
            F = flat(feature_tensor(s, np.nan_to_num(Xh), hgh))
            F = D.apply_pipeline(F, prep, patient) if prep else F
            lh = np.array([int(np.argmax(nnls(B, F[i])[0])) for i in range(n)])
            labs.append(np.array([remap.get(int(x), int(x)) for x in lh]))
        l1, l2 = labs
        full = matched[s]
        same12 = (l1 == l2) & ok
        three = same12 & (l1 == full)
        reliable[s], half_lab[s] = three, (l1, l2)
        rel_rows.append(dict(set=s, bins=bins_of(s), ari_halves_refit=float(summ.loc[s, "ari_halves"]),
                             ari_halves_fixed=ARI(l1[ok], l2[ok]), share_halves_agree=float(same12[ok].mean()),
                             share_reliable=float(three[ok].mean()), ari_vs_published=float(A[names.index(s), 0]),
                             **{f"reliable_at_margin_{m:g}": float(three[ok & (margins[s] >= m)].mean()) for m in (0.05, 0.1, 0.15, 0.2)},
                             **{f"retained_at_margin_{m:g}": float((ok & (margins[s] >= m)).mean()) for m in (0.05, 0.1, 0.15, 0.2)}))
    rel = pd.DataFrame(rel_rows)
    rel.round(3).to_csv(out / "reliability.tsv", sep="\t", index=False)

    # ---- the electrode table ---------------------------------------------------------------------------
    tab = pd.DataFrame({"patient_id": df["patient_id"], "contact_norm": df["contact_norm"], "electrode": df["electrode"],
                        "published": today, "published_margin": np.round(today_margin, 4), "sweep_refit": parts["sweep refit"],
                        "hg_resp_db": np.round(amp, 3), "halves_ok": ok, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
    for s in sets:
        tab[f"{s}_label"] = matched[s]
        tab[f"{s}_margin"] = np.round(margins[s], 4)
        tab[f"{s}_half1"], tab[f"{s}_half2"] = half_lab[s]
        tab[f"{s}_reliable"] = reliable[s]
    tab.to_csv(out / "electrodes.tsv", sep="\t", index=False)
    # the run's own reliability, beside its labels.csv: make_coverage_bundle.py carries it into the
    # cluster visualizer's contacts.json, where every chip shows the reliable share of its dots
    ps = a.publish_set if a.publish_set in sets else None
    if ps:
        relcsv = pd.DataFrame({"patient_id": df["patient_id"], "electrode": df["electrode"], "contact_norm": df["contact_norm"],
                               "set": ps, "reliable": reliable[ps].astype(int), "halves_ok": ok.astype(int),
                               "margin": np.round(margins[ps], 4), "half1": half_lab[ps][0], "half2": half_lab[ps][1],
                               "label": matched[ps], "hg_resp_db": np.round(amp, 3), "test_run": out.name})
        relcsv.to_csv(run / "reliability.csv", index=False)
        print(f"-> {run / 'reliability.csv'}  ({ps}: {int(reliable[ps][ok].sum())} of {int(ok.sum())} reliable)")

    # ================================ G1 ================================
    fig = plt.figure(figsize=(21, 10.5), dpi=150)
    gs = fig.add_gridspec(2, K, height_ratios=[1.0, 1.45], hspace=0.42, wspace=0.28, left=0.04, right=0.99, top=0.90, bottom=0.07)
    tt = (np.arange(N_TIME * N_COND) + 0.5) / N_TIME * 100
    for i in range(K):
        ax = fig.add_subplot(gs[0, i])
        ref = today_C[i].reshape(N_COND, N_TIME).ravel()
        sc = max(np.abs(ref).max(), 1e-9)
        ax.plot(tt, ref / sc, color="#111111", lw=1.0, alpha=0.35, label="published (300)")
        for s in sets:
            C = np.load(out / f"components_{s}.npy").astype(np.float64)
            raw = parts[s]
            bm, _ = LCn.match_labels(today, raw)
            inv = {}
            for j in np.unique(raw):
                inv.setdefault(int(np.bincount(bm[raw == j]).argmax()), int(j))
            if i not in inv:
                continue
            bins = bins_of(s)
            cs = hg_part(s, C, K)[inv[i]]                                   # (3, bins)
            t = ((np.arange(bins) + 0.5) / bins * 100)[None, :] + (np.arange(N_COND) * 100)[:, None]
            ax.plot(t.ravel(), cs.ravel() / max(np.abs(cs).max(), 1e-9), color=SET_COL.get(s, "k"), lw=1.1,
                    ls="--" if s.startswith("lowhg") else "-", label=s)
        for b in (100, 200):
            ax.axvline(b, color="k", lw=0.6)
        for b in range(N_COND):
            ax.axvline(b * 100 + 50, color="k", lw=0.3, ls=":")
        ax.axhline(0, color="k", lw=0.3)
        ax.set_xticks([50, 150, 250])
        ax.set_xticklabels(["audio", "picture", "reading"], fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_title(f"component {i}  (n = {int((today == i).sum())})", fontsize=9)
        if i == 0:
            ax.set_ylabel("HG course, scaled to its peak", fontsize=8)
            h0, l0 = ax.get_legend_handles_labels()
            fig.legend(h0, l0, loc="center", ncol=len(l0), fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.555))
    # (a) agreement heatmap
    ax = fig.add_subplot(gs[1, 0:2])
    im = ax.imshow(A, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.5)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if A[i, j] < 0.6 else "black")
    ax.set_title(f"(a) ARI between partitions. published vs sweep refit = the fit-to-fit floor ({A[0, 1]:.2f})", fontsize=9, loc="left")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    # (b) agreement with the published fit by margin bin
    ax = fig.add_subplot(gs[1, 2:4])
    edges = np.array([0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.45, 1.0])
    mid = 0.5 * (edges[1:] + edges[:-1])
    h, _ = np.histogram(today_margin[np.isfinite(today_margin)], bins=edges)
    ax2 = ax.twinx()
    ax2.bar(mid, h / n * 100, width=np.diff(edges) * 0.9, color="#ddd", zorder=0)
    ax2.set_ylabel("% of electrodes (grey bars)", fontsize=8, color="#666")
    ax2.tick_params(labelsize=7, colors="#666")
    for s in ["sweep refit"] + sets:
        ag = (parts[s] == today) if s == "sweep refit" else (matched[s] == today)
        vals = [ag[(today_margin >= lo) & (today_margin < hi)].mean() * 100 if ((today_margin >= lo) & (today_margin < hi)).sum() >= 10 else np.nan for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(mid, vals, marker="o", ms=3, lw=1.2, color=SET_COL.get(s, "#888"), ls="--" if s.startswith("lowhg") else "-", label=s, zorder=3)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xticks([0, 0.05, 0.1, 0.2, 0.45])
    ax.set_xticklabels(["0", ".05", ".1", ".2", ".45"], fontsize=7)
    ax.set_ylim(0, 102)
    ax.set_xlabel("margin of the published fit (top weight minus second)", fontsize=8)
    ax.set_ylabel("% with the published label", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    ax.set_title("(b) who keeps the published label: by margin, per set", fontsize=9, loc="left")
    # (c) reliabilities per set
    ax = fig.add_subplot(gs[1, 4:K])
    xs = np.arange(len(sets))
    w = 0.26
    ax.bar(xs - w, rel["ari_halves_refit"], w, color="#bbb", label="split-half ARI, each half refitted (the test)")
    ax.bar(xs, rel["ari_halves_fixed"], w, color="#4a6fa5", label="split-half ARI, components held fixed")
    ax.bar(xs + w, rel["share_reliable"], w, color="#1b7837", label="share of electrodes: full fit = half 1 = half 2")
    for i, s in enumerate(sets):
        ax.text(xs[i] + w, rel["share_reliable"][i] + 0.01, f"{rel['share_reliable'][i]:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(sets, fontsize=8)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.set_title("(c) what reproduces between the odd and the even trials", fontsize=9, loc="left")
    fig.text(0.04, 0.955, f"G1 - what changes with the grid, and what does not   (convex NMF, K = {K}, {cache['name']}, {n} electrodes; run {out.name})",
             fontsize=13, weight="bold")
    fig.text(0.04, 0.925, "Row 1: the published components with every set's matched component on top. Row 2: how the hard labels move, and why: "
                          "the published fit disagrees with its own sweep refit on the low-margin electrodes; the grids move the same electrodes.", fontsize=8.6, color="#333")
    fig.savefig(out / "G1_what_changes.png", dpi=150, facecolor="white")
    plt.close(fig)

    # ================================ G2 ================================
    s0 = a.set if a.set in sets else sets[-1]
    r0 = reliable[s0]
    m0 = margins[s0]
    lab0 = matched[s0]
    fig = plt.figure(figsize=(17, 13), dpi=150)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.0, 1.0], hspace=0.42, wspace=0.3, left=0.05, right=0.98, top=0.91, bottom=0.05)
    # row 1: margin vs amplitude
    ax = fig.add_subplot(gs[0, 0:2])
    cols = plt.get_cmap("tab10")(lab0 % 10)
    ax.scatter(m0[ok & ~r0], amp[ok & ~r0], s=9, color="#c8c8c8", lw=0, label=f"not reliable ({int((ok & ~r0).sum())})")
    ax.scatter(m0[r0], amp[r0], s=11, c=cols[r0], lw=0, label=f"reliable ({int(r0.sum())}), coloured by cluster")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xticks([0, 0.05, 0.1, 0.2, 0.4, 0.8])
    ax.set_xticklabels(["0", ".05", ".1", ".2", ".4", ".8"])
    ax.set_xlabel(f"margin ({s0})", fontsize=9)
    ax.set_ylabel("HG response, best condition (dB re baseline)", fontsize=9)
    ax.axhline(0, color="k", lw=0.4)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title(f"(a) {s0}: an electrode is reliable when the full fit and both trial halves give it the same component", fontsize=9.5, loc="left")
    ax = fig.add_subplot(gs[0, 2])
    for lab_, msk, col in (("reliable", r0, "#1b7837"), ("not reliable", ok & ~r0, "#999")):
        ax.hist(amp[msk], bins=np.linspace(-2, 8, 41), histtype="step", lw=1.5, color=col, label=f"{lab_}: median {np.median(amp[msk]):.2f} dB")
    ax.set_xlabel("HG response (dB)", fontsize=9)
    ax.set_ylabel("electrodes", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("(b) amplitude of the two groups", fontsize=9.5, loc="left")
    # row 2: the gate
    ax = fig.add_subplot(gs[1, 0:2])
    axr = ax.twinx()
    for s in sets:
        r, m = reliable[s], margins[s]
        share = [r[ok & (m >= t)].mean() * 100 if (ok & (m >= t)).sum() >= 20 else np.nan for t in THRESH]
        kept = [(ok & (m >= t)).sum() for t in THRESH]
        ax.plot(THRESH, share, color=SET_COL.get(s, "k"), lw=1.4, ls="--" if s.startswith("lowhg") else "-", label=s)
        axr.plot(THRESH, kept, color=SET_COL.get(s, "k"), lw=0.8, ls=":", alpha=0.8)
    ax.set_xlabel("margin threshold m  (keep an electrode if its margin >= m)", fontsize=9)
    ax.set_ylabel("% reliable among the kept electrodes (solid)", fontsize=9)
    axr.set_ylabel("electrodes kept (dotted)", fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7.5, frameon=False, loc="center right", ncol=1)
    ax.grid(alpha=0.3)
    ax.set_title("(c) the gate: what a margin threshold buys, per set", fontsize=9.5, loc="left")
    ax = fig.add_subplot(gs[1, 2])
    cl = np.unique(lab0)
    share_c = [r0[ok & (lab0 == c)].mean() * 100 for c in cl]
    n_c = [int((lab0 == c).sum()) for c in cl]
    ax.bar(cl, share_c, color=plt.get_cmap("tab10")(cl % 10))
    for c, sh, nn in zip(cl, share_c, n_c):
        ax.text(c, sh + 1, f"{nn}", ha="center", fontsize=7)
    ax.set_xlabel("cluster (published ids)", fontsize=9)
    ax.set_ylabel("% of its electrodes reliable", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title(f"(d) {s0}: reliability per cluster (n above)", fontsize=9.5, loc="left")
    # row 3: maps
    good = ~np.isnan(xyz).any(1)
    for j, (p, q, ti) in enumerate([(0, 1, "axial"), (0, 2, "sagittal"), (1, 2, "coronal")]):
        ax = fig.add_subplot(gs[2, j])
        m1 = good & ok & ~r0
        m2 = good & r0
        ax.scatter(xyz[m1, p], xyz[m1, q], s=7, color="#d0d0d0", lw=0)
        ax.scatter(xyz[m2, p], xyz[m2, q], s=10, c=cols[m2], lw=0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"(e) {ti}: reliable electrodes by cluster, the rest grey", fontsize=9)
    fig.text(0.05, 0.955, f"G2 - which electrodes can be trusted   ({s0}; convex NMF K = {K}; {int(r0.sum())} of {int(ok.sum())} reliable = {100 * r0[ok].mean():.0f} %)",
             fontsize=13, weight="bold")
    fig.text(0.05, 0.93, "Reliable = same component from the full fit and from each trial half, the halves assigned with the full fit's components held fixed. "
                         "Margin = top loading minus the second; amplitude = mean HG in the response half, best condition.", fontsize=8.6, color="#333")
    fig.savefig(out / "G2_reliability.png", dpi=150, facecolor="white")
    plt.close(fig)

    print(rel.round(3).to_string(index=False))
    print(f"\n-> {out / 'G1_what_changes.png'}\n-> {out / 'G2_reliability.png'}\n-> {out / 'electrodes.tsv'}, {out / 'reliability.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
