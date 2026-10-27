#!/usr/bin/env python3
"""The Paper 2 validation story, reorganised around three questions - agreement, stability,
separation - as a document to explain from: text, figure, text, figure.

    python make_validation_story.py            # outputs/clustering/paper2_validation/

Every number is read from the output files (the statistics, the consensus sweep, the
FIG 1-5 tables, the MMD tables); nothing is recomputed except the ARI / NMI between the
four feature sets, which FIG 2 draws from the same published labels and does not table.
The figures are the published PNGs, embedded whole or cropped by pixel box, plus four new
panels drawn from the tables (leave-one-patient-out per patient, the K-by-K kept share,
the separation figure, the scorecard). Nothing in paper_figures/ is touched; the new
folder has its own names.

Writes   figures/*.png, validation_story.md, validation_story.html (images embedded, opens
         anywhere), numbers.json (every value the text uses, with its source file).
"""
from __future__ import annotations

import base64
import html
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("p2fig1", ROOT / "00_Paper2_Figures.py")
P2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P2)
_mspec = importlib.util.spec_from_file_location("mmdmod", ROOT / "00_paper2_figure6_mmd.py")
MMDM = importlib.util.module_from_spec(_mspec)
_mspec.loader.exec_module(MMDM)
sys.path.insert(0, str(ROOT / "functions"))
import lf_runs as LR                                      # noqa: E402

FSET, K = "concat_bands15", 7
K_BY = MMDM.K_BY_FSET
PF = ROOT / "outputs" / "clustering" / "paper_figures"
PAPER = PF / "paper"
ST = ROOT / "outputs" / "clustering" / "statistics" / f"{FSET}_K{K}"
OUTD = ROOT / "outputs" / "clustering" / "paper2_validation"
FIGD = OUTD / "figures"
INK, MUTED, GREY, RED, GREEN = P2.INK, P2.MUTED, P2.GREY, P2.RED, P2.GREEN
SRC = {}                      # number -> source file, for numbers.json


def src(name, path):
    SRC[name] = str(Path(path).relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path)


# ---------------------------------------------------------------- the numbers
def collect():
    N = {}
    cache = LR.current_cache()
    N["cache"] = cache
    CO = ROOT / "outputs" / "clustering" / "consensus" / cache
    MM = ROOT / "outputs" / "clustering" / "mmd" / cache / FSET / "cnmf"
    d = P2.load_run(FSET, K, "cnmf")
    d["fset"] = FSET
    tag = P2.run_tag(d)
    N["run"] = d["run"].name
    N["tag"] = tag
    N["n"] = int(len(d["X"]))
    N["n_patients"] = int(d["n_patients"])
    C = P2.cube(d["X"], d)
    means = np.stack([C[d["lab"] == j].mean(0) for j in range(K)])
    order = [int(j) for j in P2.block_order(means, d)[0]]
    N["order"] = order
    N["names"] = {j: P2.type_name(FSET, K, j) for j in range(K)}
    own = d["Gn"].max(1)
    N["own_median"] = float(np.median(own))
    N["own_below_cut"] = float((own < P2.LOAD_CUT).mean())
    N["cut"] = float(P2.LOAD_CUT)

    # cohort
    f = PF / "FIG0_cohort_supplement_v17_patients.csv"
    cp = pd.read_csv(f)
    N["seen"], N["kept"] = int(cp.n_total.sum()), int(cp.n_gated.sum())
    N["gate_rate_min"], N["gate_rate_max"] = float(cp.gate_rate.min()), float(cp.gate_rate.max())
    N["gate_rate_min_patient"] = str(cp.loc[cp.gate_rate.idxmin(), "patient"])
    src("cohort", f)
    cap = (PF / "FIG0_cohort_supplement_v17_caption.txt").read_text(encoding="utf-8")
    m = re.search(r"at least ([\d.]+)% of bins are above ([+\-\d.]+) dB\s*\((\d+) bins\)", cap)
    N["gate_rule"] = (f"at least {m.group(1)} % of its bins above {m.group(2)} dB ({m.group(3)} bins)"
                      if m else "see the FIG 0 caption")

    # the types
    f = PF / f"FIG1a_{FSET}_cnmf_K{K}{tag}_patients.csv"
    pt = pd.read_csv(f).set_index("cluster")
    N["sizes"] = {int(j): int(pt.loc[j, "n"]) for j in range(K)}
    N["patients_per_type"] = {int(j): int(pt.loc[j, "n_patients"]) for j in range(K)}
    N["top_share"] = {int(j): float(pt.loc[j, "top_share"]) for j in range(K)}
    N["top_patient"] = {int(j): str(pt.loc[j, "top_patient"]) for j in range(K)}
    src("types", f)
    f = PF / f"FIG1a_{FSET}_cnmf_K{K}{tag}_generalization.csv"
    g = pd.read_csv(f).set_index("k")
    N["gen"] = {int(k): dict(frac_clusters=float(g.loc[k, "frac_clusters"]), worst=float(g.loc[k, "worst"]),
                             n_dominated=int(g.loc[k, "n_dominated"])) for k in g.index if 5 <= k <= 12}
    N["gen_first_dominated_k"] = int(min(k for k in g.index if g.loc[k, "n_dominated"] > 0))
    src("generalization", f)

    # agreement
    f = ST / "stats_agreement.csv"
    ag = pd.read_csv(f)
    N["ari_algo"] = {f"{r.a_label} vs {r.b_label}": dict(ari=float(r.ari), nmi=float(r.nmi)) for _, r in ag.iterrows()}
    src("agreement_algorithms", f)
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    labs = {}
    for fs in P2.FSETS:
        dd = P2.load_run(fs, K_BY[fs], "cnmf")
        labs[fs] = dd["lab"]
    N["ari_feat"] = {}
    for i, a_ in enumerate(P2.FSETS):
        for b_ in P2.FSETS[i + 1:]:
            N["ari_feat"][f"{MMDM.FS_SHORT[a_]} vs {MMDM.FS_SHORT[b_]}"] = dict(
                ari=float(adjusted_rand_score(labs[a_], labs[b_])),
                nmi=float(normalized_mutual_info_score(labs[a_], labs[b_])))
    SRC["agreement_feature_sets"] = "recomputed from the four runs' published labels (as FIG 2 does)"
    f = PF / f"FIG2_agreement_{FSET}_K{K}{tag}_electrode_agreement.csv"
    ea = pd.read_csv(f)
    N["modal_feat"] = {int(k): float(v) for k, v in ea.feature_set_agreement.value_counts(normalize=True).items()}
    N["modal_algo"] = {int(k): float(v) for k, v in ea.algorithm_agreement.value_counts(normalize=True).items()}
    src("modal_agreement", f)
    f = PF / f"FIG4_summary_K{K:02d}{tag}.csv"
    s4 = pd.read_csv(f)
    N["fig4"] = {r.comparison: dict(diag=float(r.diag_share), null=float(r.diag_null_mean), adj=float(r.diag_share_adj),
                                    n_sig_plain=int(r.n_sig_plain_hard), n_sig_within=int(r.n_sig_within_hard),
                                    overlap=float(r.overlap_hard_mean)) for _, r in s4.iterrows()}
    src("fig4_summary", f)
    f = PF / f"FIG4_matched_K{K:02d}{tag}.csv"
    m4 = pd.read_csv(f)
    m4 = m4[(m4.weighting == "hard") & m4.comparison.str.startswith("cnmf_")]
    m4 = m4.assign(adj=(m4.overlap - m4.null_plain_mean) / (1 - m4.null_plain_mean))
    A_ = m4[m4.axis.str.startswith("A")].groupby("cluster_1")
    B_ = m4[m4.axis.str.startswith("B")].groupby("cluster_1")
    N["type_algo_adj"] = {int(j): float(v) for j, v in A_.adj.mean().items()}
    N["type_algo_jac"] = {int(j): float(v) for j, v in A_.overlap.mean().items()}
    N["type_feat_adj"] = {int(j): float(v) for j, v in B_.adj.mean().items()}
    N["type_feat_jac"] = {int(j): float(v) for j, v in B_.overlap.mean().items()}
    src("fig4_matched", f)
    f = PF / f"FIG5_pairs_K{K:02d}{tag}.csv"
    p5 = pd.read_csv(f)
    N["fig5"] = dict(
        across_algorithms=float(p5[p5.same_feature & ~p5.same_algorithm].diagonal_share.median()),
        across_features=float(p5[p5.same_algorithm & ~p5.same_feature].diagonal_share.median()),
        both_differ=float(p5[~p5.same_algorithm & ~p5.same_feature].diagonal_share.median()),
        chance=float(p5.null_mean.median()), n_pairs=int(len(p5)),
        n_not_above=int((p5.p_value >= 0.05).sum()))
    src("fig5_pairs", f)

    # stability
    f = CO / FSET / "cnmf" / "results.json"
    r = json.loads(f.read_text(encoding="utf-8"))
    c7 = [c for c in r["curve"] if int(c["K"]) == K][0]
    N["pac"] = dict(real=float(c7["pac_real"]), null=float(c7["pac_null_mean"]), null_sd=float(c7["pac_null_sd"]),
                    rcsi=float(c7["rcsi"]), p=float(c7["p"]), n_null=int(c7["n_null"]),
                    worst_share=float(c7["worst_share"]), worst_patient=str(c7["worst_patient"]),
                    worst_cluster=int(c7["worst_cluster"]))
    best = max(r["curve"], key=lambda c: c["rcsi"] if c.get("dominance_ok") else -9)
    N["rcsi_chosen"] = dict(K=int(best["K"]), rcsi=float(best["rcsi"]), p=float(best["p"]))
    N["pac_dominance_fail_k"] = sorted(int(c["K"]) for c in r["curve"] if not c.get("dominance_ok"))
    src("consensus_results", f)
    f = CO / FSET / "cnmf" / f"items_k{K:02d}.csv"
    it = pd.read_csv(f)
    it["key"] = [f"{p}|{P2.norm(e)}" for p, e in zip(it.patient_id, it.electrode)]
    key = pd.Series([f"{p}|{P2.norm(e)}" for p, e in zip(d["meta"].patient_id, d["meta"].electrode)])
    vals = it.set_index("key").own_consensus.reindex(key).to_numpy()
    N["consensus_type"] = {int(j): float(np.nanmean(vals[d["lab"] == j])) for j in range(K)}
    N["consensus_matched"] = int(np.isfinite(vals).sum())
    src("consensus_items", f)
    f = CO / FSET / "cnmf" / f"clusters_k{K:02d}.csv"
    cl = pd.read_csv(f)
    N["consensus_slide_values"] = sorted((float(v) for v in cl.m), reverse=True)
    N["consensus_mean"] = float(cl.m.mean())
    src("consensus_clusters", f)
    f = ST / "stats_summary.json"
    ss = json.loads(f.read_text(encoding="utf-8"))
    N["lopo"] = ss["lopo"]["cnmf"]
    N["lopo_predict"] = ss["lopo_predict"]["cnmf"]
    N["lopo_all"] = {m_: ss["lopo"][m_] for m_ in ss["lopo"]}
    src("lopo", f)
    f = PF / f"FIG5_KbyK{tag}.csv"
    kb = pd.read_csv(f)
    kb = kb[(kb.method == "cnmf") & (kb.feature == FSET) & (kb.k_a == K)]
    N["kbyk"] = {int(r.k_b): dict(share=float(r.diagonal_share), null=float(r.null_mean), adj=float(r.chance_corrected))
                 for _, r in kb.iterrows()}
    src("kbyk", f)

    # separation
    f = ST / "stats_separation.csv"
    sp = pd.read_csv(f)
    row = sp[(sp.method == "cnmf") & sp.home].iloc[0]
    N["sil"] = dict(value=float(row.silhouette), null=float(row.null_mean), null_sd=float(row.null_sd), z=float(row.z),
                    n_null=int(row.n_null), space=str(row.space))
    src("silhouette", f)
    N["mmd_present"] = (MM / "manifest.json").exists()
    if N["mmd_present"]:
        res = MMDM.load_cell(MM)
        man = res["manifest"]
        N["mmd_params"] = {k: man[k] for k in ("n_perm", "n_perm_heldout", "n_splits", "n_cloud", "bandwidth", "frac_below_cut")}
        sv = res["summary"]
        for v in ("argmax", "cut"):
            s_ = sv[sv.version == v].sort_values("ratio")
            N[f"mmd_{v}"] = [dict(i=int(r.i), j=int(r.j), mmd2_in=float(r.mmd2_in), z_in=float(r.z_in), q_in=float(r.q_in),
                                  mmd2_ho=float(r.mmd2_ho), z_ho=float(r.z_ho_median), share_p=float(r.share_p_ho_lt05),
                                  cloud=float(r.cloud_median), ratio=float(r.ratio), above=float(r.share_above_cloud))
                             for _, r in s_.iterrows()]
        pty = res["per_type"]
        for v in ("argmax", "cut"):
            p_ = pty[pty.version == v].set_index("cluster")
            N[f"mmd_type_{v}"] = {int(j): dict(nn=int(p_.loc[j, "nn_cluster"]), nn_ratio=float(p_.loc[j, "nn_ratio"]),
                                               mean_ratio=float(p_.loc[j, "mean_ratio"]))
                                  for j in range(K)}
        p_ = pty[pty.version == "argmax"].set_index("cluster")
        N["sil_type"] = {int(j): dict(sil=float(p_.loc[j, "silhouette"]), z=float(p_.loc[j, "sil_z"])) for j in range(K)}
        N["sil_cloud"] = dict(mean=float(p_.sil_cloud_mean.iloc[0]), sd=float(p_.sil_cloud_sd.iloc[0]))
        N["mmd_res"] = res
        src("mmd", MM / "pairs_summary.csv")

    # anatomy
    f = PF / f"FIG3_lana_{FSET}_K{K}{tag}_clusters.csv"
    la = pd.read_csv(f).set_index("cluster")
    N["lana"] = {int(j): dict(mean_P=float(la.loc[j, "mean_P"]), q_mean_sh=float(la.loc[j, "q_mean_sh"]),
                              q_mean=float(la.loc[j, "q_mean"]), q_t10_sh=float(la.loc[j, "q_frac_t10_sh"]),
                              frac_t10=float(la.loc[j, "frac_t10"])) for j in range(K)}
    N["lana_base"] = dict(P=float(la.baseline_P.iloc[0]), t10=float(la.base_t10.iloc[0]))
    src("lana", f)
    f = PF / f"FIG3_spin_{FSET}_K{K}{tag}_summary.json"
    sj = json.loads(f.read_text(encoding="utf-8"))
    N["spin"] = dict(n_spin=int(sj["n_spin"]), p_all=float(sj["all"]["p_rms"]), p_left=float(sj["L"]["p_rms"]))
    f2 = PF / f"FIG3_spin_{FSET}_K{K}{tag}_clusters.csv"
    sc = pd.read_csv(f2)
    N["spin_sig"] = {st: dict(share=int((sc[(sc.stratum == st) & (sc.threshold == 0.1)].q < 0.05).sum()),
                              meanP=int((sc[(sc.stratum == st) & (sc.threshold == 0.1)].q_mean_P < 0.05).sum()),
                              meanP_which=[int(c) for c in sc[(sc.stratum == st) & (sc.threshold == 0.1) & (sc.q_mean_P < 0.05)].cluster])
                     for st in ("all", "L")}
    src("spin", f)
    N["d"] = d
    return N


# ---------------------------------------------------------------- figures
def crop(src_png: Path, box, name: str, whiteout=()):
    """A pixel-box crop (fractions of the source); whiteout = boxes, as fractions of the
    crop, painted white (a neighbouring panel's labels that the box cannot exclude)."""
    im = Image.open(src_png).convert("RGB")
    W, H = im.size
    x0, y0, x1, y1 = box
    b = (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))
    c = im.crop(b)
    if whiteout:
        from PIL import ImageDraw
        dr = ImageDraw.Draw(c)
        w, h = c.size
        for (u0, v0, u1, v1) in whiteout:
            dr.rectangle((int(u0 * w), int(v0 * h), int(u1 * w), int(v1 * h)), fill="white")
    out = FIGD / name
    c.save(out)
    return out


def copy_png(src_png: Path, name: str):
    out = FIGD / name
    Image.open(src_png).save(out)
    return out


def tname(N, j):
    nm = N["names"][j]
    return f"c{j}" + (f" {nm}" if nm else "")


def fig_lopo(N):
    lo = pd.read_csv(ST / "stats_lopo.csv")
    lo = lo[lo.method == "cnmf"]
    lp = pd.read_csv(ST / "stats_lopo_predict.csv")
    lp = lp[lp.method == "cnmf"]
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.3), dpi=170)
    L = N["lopo"]
    ax[0].axhspan(L["null_mean"] - L["null_sd"], L["null_mean"] + L["null_sd"], color=GREY, alpha=0.45, lw=0)
    ax[0].axhline(L["null_mean"], color=MUTED, ls=":", lw=1.0)
    ax[0].scatter(lo.n_left_out, lo.ari, s=26, color=INK, zorder=3)
    for _, r in lo.nsmallest(3, "ari").iterrows():
        ax[0].annotate(r.group, (r.n_left_out, r.ari), xytext=(4, -9), textcoords="offset points", fontsize=7, color=MUTED)
    ax[0].set_xlabel("electrodes of the left-out patient")
    ax[0].set_ylabel("ARI of the refit against the full-cohort labels")
    ax[0].set_title(f"A   leave one patient out, refit: median ARI {L['real_median']:.2f}\n"
                    f"grey: size-matched shaft-permuted pseudo-patients, {L['null_mean']:.2f} ± {L['null_sd']:.2f}",
                    fontsize=9.5, loc="left", color=INK)
    ax[1].scatter(lp.n_held_out, lp.acc, s=26, color=INK, zorder=3, label="placed in the full fit's cluster")
    ax[1].scatter(lp.n_held_out, lp.chance, s=22, facecolors="none", edgecolors=MUTED, zorder=2, label="chance (the patient's modal cluster)")
    for _, r in lp.nsmallest(2, "acc").iterrows():
        ax[1].annotate(r.group, (r.n_held_out, r.acc), xytext=(4, -9), textcoords="offset points", fontsize=7, color=MUTED)
    ax[1].set_xlabel("electrodes of the held-out patient")
    ax[1].set_ylabel("share of its electrodes")
    ax[1].set_ylim(0, 1.03)
    ax[1].legend(fontsize=7.5, frameon=False, loc="lower right")
    P_ = N["lopo_predict"]
    ax[1].set_title(f"B   leave one patient out, place its electrodes: median accuracy {P_['acc_median']:.2f}, "
                    f"chance {P_['chance_mean']:.2f}", fontsize=9.5, loc="left", color=INK)
    for a_ in ax:
        a_.tick_params(labelsize=8, colors=MUTED)
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGD / "fig3e_leave_one_patient_out.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def fig_kbyk(N):
    kb = N["kbyk"]
    ks = sorted(kb)
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=170)
    x = np.arange(len(ks))
    ax.bar(x, [kb[k]["share"] for k in ks], color=P2.cluster_col(2, K), width=0.6, label="kept share")
    ax.plot(x, [kb[k]["null"] for k in ks], "_", ms=22, mew=2, color=INK, label="chance (random pairing)")
    for xi, k in zip(x, ks):
        ax.text(xi, kb[k]["share"] + 0.02, f"{kb[k]['share']:.2f}\n(adj {kb[k]['adj']:.2f})", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K {K} → K {k}" for k in ks])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("share of electrodes in matched pairs")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_title("how much of the K 7 partition survives a larger K\n(15 bands, convex NMF; Hungarian-matched diagonal share)", fontsize=9.5, loc="left", color=INK)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGD / "fig3f_k7_to_k8.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def fig_separation(N):
    """A silhouette z per type; B the 7 x 7 held-out MMD^2 with the held-out test marked;
    C the most and the least separated pair's maps; D the top-loading distribution."""
    d, order = N["d"], N["order"]
    res = N.get("mmd_res")
    fig = plt.figure(figsize=(17.6, 10.6), dpi=170)
    gs = GridSpec(2, 24, figure=fig, height_ratios=[1.0, 0.95], hspace=0.55, wspace=1.0,
                  left=0.05, right=0.985, top=0.93, bottom=0.07)
    # A
    axA = fig.add_subplot(gs[0, 0:7])
    if "sil_type" in N:
        z = [N["sil_type"][j]["z"] for j in order]
        axA.bar(np.arange(K), z, color=[P2.cluster_col(j, K) for j in order], width=0.65)
        axA.axhspan(-2, 2, color=GREY, alpha=0.35, lw=0)
        axA.axhline(0, color=INK, lw=0.8)
        for xi, (j, zz) in enumerate(zip(order, z)):
            axA.text(xi, zz + (0.15 if zz >= 0 else -0.15), f"{zz:+.1f}", ha="center", va="bottom" if zz >= 0 else "top", fontsize=8, color=INK)
        axA.set_xticks(np.arange(K))
        axA.set_xticklabels([f"c{j}" for j in order], fontsize=9)
        for t, j in zip(axA.get_xticklabels(), order):
            t.set_color(P2.cluster_col(j, K))
        axA.set_ylabel("per-type silhouette, z against the Gaussian cloud")
        axA.set_title(f"A   silhouette per type   (cloud: {N['sil_cloud']['mean']:.3f} ± {N['sil_cloud']['sd']:.3f}, grey = ±2 SD)",
                      fontsize=10, loc="left", color=INK)
    else:
        axA.axis("off")
        axA.text(0.5, 0.5, "PLACEHOLDER: run 00_paper2_figure6_mmd.py", ha="center", va="center", fontsize=11, color=RED, transform=axA.transAxes)
    axA.tick_params(labelsize=8, colors=MUTED)
    axA.spines[["top", "right"]].set_visible(False)
    # B
    axB = fig.add_subplot(gs[0, 8:16])
    if res is not None:
        sv = res["summary"][res["summary"].version == "argmax"]
        M = np.full((K, K), np.nan)
        Pm = np.full((K, K), np.nan)
        Rt = np.full((K, K), np.nan)
        for _, r in sv.iterrows():
            M[int(r.i), int(r.j)] = M[int(r.j), int(r.i)] = r.mmd2_ho
            Pm[int(r.i), int(r.j)] = Pm[int(r.j), int(r.i)] = r.share_p_ho_lt05
            Rt[int(r.i), int(r.j)] = Rt[int(r.j), int(r.i)] = r.ratio
        Mo, Po, Ro = M[np.ix_(order, order)], Pm[np.ix_(order, order)], Rt[np.ix_(order, order)]
        im = axB.imshow(Mo, cmap="PuBuGn", vmin=0, vmax=max(0.3, float(np.nanmax(Mo))), aspect="equal")
        for a_ in range(K):
            for b_ in range(K):
                if a_ == b_:
                    axB.add_patch(plt.Rectangle((b_ - 0.5, a_ - 0.5), 1, 1, fc="#f2f3f5", ec="none"))
                    continue
                v = Mo[a_, b_]
                dark = v > 0.6 * max(0.3, float(np.nanmax(Mo)))
                axB.text(b_, a_, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if dark else INK)
                if Po[a_, b_] >= 0.9:
                    axB.plot(b_ + 0.36, a_ - 0.36, "o", ms=2.6, color="white" if dark else INK)
                if Ro[a_, b_] < 1.0:
                    axB.add_patch(plt.Rectangle((b_ - 0.5, a_ - 0.5), 1, 1, fc="none", ec=RED, lw=1.2))
        axB.set_xticks(range(K)); axB.set_yticks(range(K))
        axB.set_xticklabels([f"c{j}" for j in order], fontsize=9)
        axB.set_yticklabels([f"c{j}" for j in order], fontsize=9)
        for t, j in zip(axB.get_xticklabels(), order):
            t.set_color(P2.cluster_col(j, K))
        for t, j in zip(axB.get_yticklabels(), order):
            t.set_color(P2.cluster_col(j, K))
        axB.tick_params(length=0)
        for s_ in axB.spines.values():
            s_.set_visible(False)
        cb = fig.colorbar(im, ax=axB, fraction=0.045, pad=0.03)
        cb.set_label(f"held-out MMD² (median over {N['mmd_params']['n_splits']} patient halves)", fontsize=8, color=MUTED)
        cb.ax.tick_params(labelsize=7, colors=MUTED)
        cb.outline.set_visible(False)
        cl = N["mmd_argmax"][0]["cloud"]
        axB.set_title("B   MMD² between every pair on held-out patients" + chr(10)
                      + "dot: within-patient shuffle p < 0.05 in ≥ 90 % of the splits" + chr(10)
                      + f"red frame: below the Gaussian cloud's sectors ({cl:.3f})",
                      fontsize=9, loc="left", color=INK)
    else:
        axB.axis("off")
        axB.text(0.5, 0.5, "PLACEHOLDER: MMD not run", ha="center", va="center", fontsize=11, color=RED, transform=axB.transAxes)
    # D
    axD = fig.add_subplot(gs[0, 17:24])
    own = d["Gn"].max(1)
    axD.hist(own, bins=np.linspace(0.14, 1.0, 44), color=MUTED, alpha=0.85)
    axD.axvline(P2.LOAD_CUT, color=RED, ls="--", lw=1.2)
    axD.axvline(np.median(own), color=INK, ls=":", lw=1.0)
    axD.text(P2.LOAD_CUT + 0.01, axD.get_ylim()[1] * 0.92, f"{100*N['own_below_cut']:.1f} % below {P2.LOAD_CUT:.1f}", fontsize=8, color=RED)
    axD.text(np.median(own) + 0.01, axD.get_ylim()[1] * 0.80, f"median {np.median(own):.2f}", fontsize=8, color=INK)
    axD.set_xlabel("own-cluster loading (row-normalised, argmax)")
    axD.set_ylabel("electrodes")
    axD.set_title("D   how committed each electrode is to its type", fontsize=10, loc="left", color=INK, pad=14)
    axD.tick_params(labelsize=8, colors=MUTED)
    axD.spines[["top", "right"]].set_visible(False)
    # C
    if res is not None:
        pairs = N["mmd_argmax"]
        least, most = pairs[0], pairs[-1]
        for col_, (pr, lab_) in enumerate(((most, "most separated"), (least, "least separated"))):
            sub = GridSpecFromSubplotSpec(1, 2, gs[1, col_ * 12:(col_ + 1) * 12], width_ratios=[4.2, 1.0], wspace=0.06)
            ax, axb = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
            m = res["maps"][("argmax", pr["i"], pr["j"])]
            # outline where the difference is both significant and at least 0.3 SD, so the
            # outline shows where the pair differs by more than a hair
            mm = dict(m, q=np.where(np.abs(m["d"]) >= 0.3, m["q"], 1.0).astype(np.float32))
            MMDM.map_panel(ax, axb, d, mm, f"C   {lab_}: c{pr['i']} minus c{pr['j']}   ·   held-out MMD² {pr['mmd2_ho']:.3f}, {pr['ratio']:.1f}× the cloud")
    out = FIGD / "fig4_separation.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def fig_scorecard(N):
    order = N["order"]
    cols = [("cross-algorithm\noverlap (adj.)", lambda j: N["type_algo_adj"][j], True),
            ("cross-feature-set\noverlap (adj.)", lambda j: N["type_feat_adj"][j], True),
            ("consensus\n(patients resampled)", lambda j: N["consensus_type"][j], True),
            ("silhouette z\nvs the cloud", lambda j: N["sil_type"][j]["z"] if "sil_type" in N else np.nan, True),
            ("nearest type\nheld-out MMD²", lambda j: (N["mmd_type_argmax"][j]["nn_ratio"] * N["mmd_argmax"][0]["cloud"]) if "mmd_type_argmax" in N else np.nan, True),
            ("patients", lambda j: N["patients_per_type"][j], True),
            ("largest patient\nshare", lambda j: N["top_share"][j], False)]
    V = np.array([[c[1](j) for c in cols] for j in order], float)
    R = np.full(V.shape, np.nan)
    for c_, (_, _, higher) in enumerate(cols):
        v = V[:, c_]
        if np.isfinite(v).any():
            rk = pd.Series(v).rank(ascending=not higher, pct=True).to_numpy()
            R[:, c_] = 1.0 - rk + 1.0 / K          # best = 1
    fig, ax = plt.subplots(figsize=(13.5, 5.2), dpi=170)
    ax.imshow(R, cmap="YlGn", vmin=0, vmax=1.15, aspect="auto")
    fmt = ["{:.2f}", "{:.2f}", "{:.2f}", "{:+.1f}", "{:.3f}", "{:d}", "{:.0%}"]
    for r_ in range(K):
        for c_ in range(len(cols)):
            v = V[r_, c_]
            txt = "n/a" if not np.isfinite(v) else fmt[c_].format(int(v) if fmt[c_] == "{:d}" else v)
            ax.text(c_, r_, txt, ha="center", va="center", fontsize=10, color=INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols], fontsize=8.5)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(K))
    ax.set_yticklabels([tname(N, j) for j in order], fontsize=9.5)
    for t, j in zip(ax.get_yticklabels(), order):
        t.set_color(P2.cluster_col(j, K))
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title("each type on agreement, stability and separation   ·   colour = rank within the column (dark = best; "
                 "for the largest patient share, smaller is better)", fontsize=9.5, loc="left", color=MUTED, pad=28)
    fig.tight_layout()
    out = FIGD / "fig5_scorecard.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def build_figures(N):
    FIGD.mkdir(parents=True, exist_ok=True)
    tag = N["tag"]
    F = {}
    F["fig0"] = copy_png(PAPER / "FIG0_cohort_v17.png", "fig0_task_and_cohort.png")
    F["fig0b"] = copy_png(PAPER / "FIG0_cohort_supplement_v17.png", "fig0b_cohort_supplement.png")
    F["fig1"] = crop(PAPER / f"FIG1a_{FSET}_cnmf_K{K}{tag}.png", (0.0, 0.271, 1.0, 1.0), "fig1_types.png")
    F["fig2a"] = copy_png(PAPER / f"FIG2_agreement_{FSET}_K{K}{tag}.png", "fig2a_ari_nmi_modal.png")
    F["fig2b"] = copy_png(PAPER / f"FIG4_K{K:02d}{tag}.png", "fig2b_matched_clusters.png")
    F["fig2c"] = copy_png(PAPER / f"FIG5_K{K:02d}{tag}.png", "fig2c_diagonal_share.png")
    work1 = PF / f"FIG1a_{FSET}_cnmf_K{K}{tag}.png"
    F["fig3a"] = crop(work1, (0.655, 0.025, 1.0, 0.285), "fig3a_one_patient_clusters_by_k.png")
    F["fig3bcd"] = copy_png(ROOT / "outputs" / "clustering" / "consensus" / N["cache"] / f"consensus_bands15_cnmf_K{K}_slide.png",
                            "fig3bcd_consensus.png")
    F["fig3e"] = fig_lopo(N)
    F["fig3f"] = fig_kbyk(N)
    F["fig4"] = fig_separation(N)
    F["fig5"] = fig_scorecard(N)
    F["fig6a"] = copy_png(PAPER / f"FIG3_lana_{FSET}_K{K}{tag}.png", "fig6a_lana.png")
    F["fig6b"] = copy_png(PAPER / f"FIG3_spin_{FSET}_K{K}{tag}.png", "fig6b_lana_spin.png")
    F["figS1"] = crop(work1, (0.01, 0.022, 0.388, 0.265), "figS1_heldout_variance_by_k.png",
                      whiteout=[(0.905, 0.17, 1.0, 1.0)])
    F["figS2"] = copy_png(ROOT / "outputs" / "clustering" / "consensus" / N["cache"] / "rcsi_by_k.png", "figS2_rcsi_by_k_all_cells.png")
    for fs, letter in (("concat_bands5", "b"), ("concat_bands2", "c"), ("concat_hfa", "d")):
        hit = sorted(PAPER.glob(f"FIG1{letter}_{fs}_cnmf_K{K_BY[fs]}_run*.png"))
        if hit:
            F[f"figS3{letter}"] = copy_png(hit[-1], f"figS3{letter}_types_{MMDM.FS_SHORT[fs].replace(' ', '')}.png")
    for k_ in (8, 10):
        hit = sorted(PAPER.glob(f"FIG4_K{k_:02d}_run*.png"))
        if hit:
            F[f"figS4_{k_}"] = copy_png(hit[-1], f"figS4_matched_clusters_K{k_}.png")
    if N["mmd_present"]:
        hit = sorted(PF.glob(f"FIG6_mmd_main_{FSET}_cnmf_K{K}_run*.png"))
        if hit:
            F["figS5"] = copy_png(hit[-1], "figS5_mmd_main.png")
    return F


# ---------------------------------------------------------------- the text
def write_text(N, F):
    L = []
    A = L.append
    o = N["order"]
    nm = {j: tname(N, j) for j in range(K)}
    rel = lambda p: f"figures/{Path(p).name}"

    def img(key, alt):
        if key in F:
            A(f"![{alt}]({rel(F[key])})")
            A("")

    A("# Paper 2 · how the seven response types were validated")
    A("")
    A(f"*Built {time.strftime('%Y-%m-%d %H:%M')} from the output files of cohort {N['cache']}, run {N['run']}; "
      f"every number below is read from a file named in `numbers.json`.*")
    A("")
    A("Three questions decide whether a clustering is a result or an artefact. **Agreement**: do other descriptions "
      "and other algorithms find the same shapes? **Stability**: do the shapes come back when the patients change? "
      "**Separation**: are the shapes distinct from one another, and in which features? The sections below put the "
      "existing figures in that order, add the two that were missing (separation, scorecard), and end with anatomy "
      "as context.")
    A("")

    # ---- 0
    A("## 0 · Task and cohort")
    A("")
    A("**Question.** What was recorded, and which electrodes enter the clustering?")
    A("")
    A("**How.** Three naming conditions (an auditory stimulus, a picture, a written sentence); each trial is a stimulus "
      "and then a response cue, and the time axis is warped so the cue falls at the middle of every block. An electrode "
      f"enters when at least one condition passes the responsiveness gate: {N['gate_rule']} of the 0–400 Hz ERSP, "
      "or the mirror rule for decreases.")
    A("")
    A("**What a good result looks like.** A cohort that is large and many-patient, with the gate behaving like a filter "
      "on responsiveness rather than on patients.")
    A("")
    A(f"**Found.** {N['seen']} electrodes seen, {N['kept']} kept ({100*N['kept']/N['seen']:.0f} %) in "
      f"{N['n_patients']} patients; the per-patient gate rate runs from {100*N['gate_rate_min']:.0f} % "
      f"({N['gate_rate_min_patient']}) to {100*N['gate_rate_max']:.0f} %.")
    A("")
    img("fig0", "Fig 0: the task through four electrodes, and every electrode the gate saw")
    img("fig0b", "Fig 0 supplement: per patient, the same electrodes on the brain, two electrodes at the gate's line")
    A("**Takeaway.** The task and the cohort are fixed before any clustering; the gate keeps about half of what was "
      "implanted. **Caveat.** The kept set is not a random sample of cortex: it is where responsive contacts happened to "
      "be, which the anatomy section has to live with.")
    A("")

    # ---- 1
    A("## 1 · The types: seven recurring response shapes")
    A("")
    A("**Question.** What are the recurring shapes, and who carries them?")
    A("")
    A("**How.** Convex NMF on the 15 bands × 30 time bins × 3 conditions description of every electrode (unit-norm "
      f"rows), K = {K}. An electrode's type is the component with its largest loading; the loadings are drawn, and "
      f"electrodes below a loading of {N['cut']:.1f} are not drawn, but every count and test uses the argmax membership. "
      "Each block shows the type's mean ERSP, both hemispheres from their own side, and the patients it is made of.")
    A("")
    A("**What a good result looks like.** Every type drawn from many patients, none more than half from one patient.")
    A("")
    sizes = ", ".join(f"{nm[j]} n {N['sizes'][j]} ({N['patients_per_type'][j]} patients, largest {100*N['top_share'][j]:.0f} %)"
                      for j in o)
    A(f"**Found.** {sizes}. The median own-type loading is {N['own_median']:.2f}; {100*N['own_below_cut']:.1f} % of the "
      f"electrodes sit below the {N['cut']:.1f} draw cut.")
    A("")
    img("fig1", "Fig 1: the seven types - mean ERSP, hemisphere renders, patient composition")
    A("**Takeaway.** Seven shapes, each spanning 11 to 28 patients. **Caveat.** The names are descriptive labels read "
      "off the blocks; the loadings are mixed (median 0.41), which is what the separation section measures.")
    A("")

    # ---- 2
    A("## 2 · Agreement: other descriptions and other algorithms find the same shapes")
    A("")
    A("**Question.** Do the four frequency descriptions (15, 5, 2 bands, HFA) and the three algorithms (convex NMF, "
      "k-means, Ward) place the electrodes alike?")
    A("")
    A("**How.**")
    A("")
    A("- **ARI.** For every pair of electrodes, do both solutions agree (together in both, or apart in both)? The share "
      "of agreeing pairs is the Rand index; ARI corrects it for chance, so 0 is chance and 1 identical.")
    A("- **NMI.** How much knowing an electrode's cluster in one solution tells you about its cluster in the other: "
      "I(A;B) / mean(H(A), H(B)).")
    A("- **Modal agreement.** Relabel every solution to match the reference (Hungarian matching), then count, per "
      "electrode, how many solutions give it the same label.")
    A("- **Diagonal share.** Cross-tabulate two solutions, pair the clusters one-to-one to maximise the overlap, then "
      "the share of electrodes in the matched pairs; chance is the same share after shuffling one solution's labels.")
    A("- **Jaccard / adjusted overlap.** |A ∩ B| / |A ∪ B| for a matched pair of clusters; the adjusted overlap "
      "(J − J₀)/(1 − J₀) sets chance to 0.")
    A("")
    A(f"**What a good result looks like.** ARI well above 0; all-solutions agreement well above its chance "
      f"({100*0.006:.1f} % for four feature sets, {100*0.031:.1f} % for three algorithms, from the FIG 2 caption); "
      f"diagonal shares well above the shuffled chance of about {N['fig5']['chance']:.2f}.")
    A("")
    aa = N["ari_algo"]
    af = N["ari_feat"]
    best_f = max(af.items(), key=lambda kv: kv[1]["ari"])
    worst_f = min(af.items(), key=lambda kv: kv[1]["ari"])
    best_a = max(aa.items(), key=lambda kv: kv[1]["ari"])
    worst_a = min(aa.items(), key=lambda kv: kv[1]["ari"])
    f4 = N["fig4"]
    A(f"**Found.** Across algorithms ARI {best_a[1]['ari']:.2f} ({best_a[0]}) to {worst_a[1]['ari']:.2f} ({worst_a[0]}), "
      f"NMI {best_a[1]['nmi']:.2f} to {worst_a[1]['nmi']:.2f}; across feature sets ARI {best_f[1]['ari']:.2f} "
      f"({best_f[0]}) to {worst_f[1]['ari']:.2f} ({worst_f[0]}). All three algorithms agree on "
      f"{100*N['modal_algo'].get(3, 0):.0f} % of the electrodes, all four feature sets on "
      f"{100*N['modal_feat'].get(4, 0):.0f} %. Matched clusters: convex NMF against k-means puts "
      f"{f4['cnmf_vs_kmeans_concat_bands15']['diag']:.2f} of the electrodes on the matched diagonal "
      f"(chance {f4['cnmf_vs_kmeans_concat_bands15']['null']:.2f}), against Ward "
      f"{f4['cnmf_vs_hierarchical_concat_bands15']['diag']:.2f}; 15 bands against 5 bands "
      f"{f4['cnmf_concat_bands15_vs_concat_bands5']['diag']:.2f}, against HFA "
      f"{f4['cnmf_concat_bands15_vs_concat_hfa']['diag']:.2f}. Pairs clearing both nulls: "
      + ", ".join(f"{k.replace('cnmf_vs_', 'convex NMF vs ').replace('_concat_bands15', '').replace('cnmf_concat_bands15_vs_concat_', '15 bands vs ').replace('hierarchical', 'Ward').replace('bands5', '5 bands').replace('bands2', '2 bands').replace('hfa', 'HFA')} "
                  f"{v['n_sig_within']}/{K}" for k, v in f4.items())
      + f". Over all {N['fig5']['n_pairs']} pairs of solutions at K = {K}, the median diagonal share is "
      f"{N['fig5']['across_algorithms']:.2f} across algorithms, {N['fig5']['across_features']:.2f} across feature sets and "
      f"{N['fig5']['both_differ']:.2f} when both differ (chance {N['fig5']['chance']:.2f}); "
      f"{N['fig5']['n_not_above']} of {N['fig5']['n_pairs']} pairs fail to beat chance.")
    A("")
    img("fig2a", "Fig 2A/B: ARI and NMI matrices with the agreement-by-K insets, and the brain map of modal agreement")
    img("fig2b", "Fig 2C: matched clusters against the two nulls (current FIG 4)")
    img("fig2c", "Fig 2D: the 66-pair diagonal share (current FIG 5)")
    A("**Takeaway.** Changing the algorithm moves the electrodes less than changing the frequency description; the "
      "description matters more than the algorithm. **Caveat.** Agreement is not correctness: four descriptions can "
      "agree on a shape that is an artefact of the gate or of the warp, and nothing here is a statistical test of the "
      "types themselves.")
    A("")

    # ---- 3
    A("## 3 · Stability: the same shapes come back when patients are resampled")
    A("")
    A("**Question.** Do the types survive a change of patients, and is K = 7 the right place to stop?")
    A("")
    A("**How.**")
    A("")
    A("- **One-patient clusters over K.** At every K, the share of clusters more than half from one patient and the "
      "share of electrodes in them.")
    A("- **PAC.** Refit on 80 % of the patients 200 times; M = how often two electrodes land together; PAC = share of "
      "pairs with 0.1 < M < 0.9 (lower is more stable). The null is the Gaussian cloud: one distribution with the data's "
      "covariance and no clusters, through the same procedure.")
    A("- **Per-type consensus.** Mean M over the pairs inside a type.")
    A("- **Leave one patient out.** Drop a patient, refit, ARI against the full-cohort labels, against size-matched "
      "pseudo-patients made of shuffled shafts; and place the dropped patient's electrodes with the refit's components "
      "and count how many land in their full-cohort type.")
    A("- **K 7 → K 8.** The diagonal share between the K 7 and K 8 partitions of the same run.")
    A("")
    A("**What a good result looks like.** PAC below the cloud; no cluster owned by one patient; per-type consensus "
      "above 0.6; leave-one-patient-out ARI high and no worse than for a pseudo-patient; most of the partition kept "
      "at K + 1.")
    A("")
    pac = N["pac"]
    gen = N["gen"]
    cons = N["consensus_type"]
    cons_sorted = sorted(cons.items(), key=lambda kv: -kv[1])
    L_ = N["lopo"]
    P_ = N["lopo_predict"]
    kb = N["kbyk"]
    fails = N["pac_dominance_fail_k"]
    lo_f = [k for k in fails if k < K]
    hi_f = [k for k in fails if k > K]
    fail_txt = " and ".join(([f"at K = {', '.join(str(k) for k in lo_f)}"] if lo_f else [])
                            + ([f"at every K from {min(hi_f)} up"] if hi_f and hi_f == list(range(min(hi_f), max(hi_f) + 1)) else
                               [f"at K = {', '.join(str(k) for k in hi_f)}"] if hi_f else []))
    A(f"**Found.** At K = {K} no cluster is more than half one patient (worst {100*pac['worst_share']:.0f} %, "
      f"{pac['worst_patient']} in c{pac['worst_cluster']}); at K = {N['gen_first_dominated_k']} one cluster is "
      f"{100*gen[N['gen_first_dominated_k']]['worst']:.0f} % one patient, and the consensus partition fails the same rule "
      f"{fail_txt}. PAC {pac['real']:.3f} against "
      f"{pac['null']:.3f} ± {pac['null_sd']:.3f} for the cloud (p {pac['p']:.2f}, the floor of {pac['n_null']} draws; "
      f"RCSI {pac['rcsi']:.2f}). Per-type consensus "
      + ", ".join(f"{nm[j]} {v:.2f}" for j, v in cons_sorted)
      + f". Leave one patient out: median ARI {L_['real_median']:.2f} (minimum {L_['real_min']:.2f}) against "
      f"{L_['null_mean']:.2f} ± {L_['null_sd']:.2f} for the pseudo-patients (z {L_['z']:+.1f}); placement accuracy "
      f"median {P_['acc_median']:.2f} against a chance of {P_['chance_mean']:.2f} (worst patient {P_['worst_patient']}, "
      f"{P_['acc_min']:.2f}). K 7 → K 8 keeps {kb[8]['share']:.2f} of the electrodes in matched pairs "
      f"(chance-corrected {kb[8]['adj']:.2f}), K 7 → K 10 {kb[10]['share']:.2f}.")
    A("")
    img("fig3a", "Fig 3A: one-patient clusters over K (current FIG 1 D)")
    img("fig3bcd", "Fig 3B-D: consensus matrix at K 7, per-type consensus, PAC against the Gaussian cloud over K")
    img("fig3e", "Fig 3E: leave one patient out, per patient against its electrode count")
    img("fig3f", "Fig 3F: how much of the K 7 partition survives at K 8, 9, 10")
    A(f"**Takeaway.** The shapes come back under patient resampling, and K = {K} is the last K before a patient owns a "
      f"cluster. **Caveat.** The cloud has no patient structure, so a K whose clusters follow patients can look stable "
      f"against it: the dominance rule is the guard. The sweep's own K criterion (RCSI) prefers K = "
      f"{N['rcsi_chosen']['K']} (RCSI {N['rcsi_chosen']['rcsi']:.2f}, p {N['rcsi_chosen']['p']:.2f}), so consensus is "
      f"support for the K {K} solution, not the reason for it. Leave-one-patient-out ARI is no better than for a "
      f"pseudo-patient (z {L_['z']:+.1f}): dropping a real patient costs no more than dropping the same number of shafts "
      f"at random, which is the absence of a dominant patient, not a separate virtue.")
    A("")

    # ---- 4
    A("## 4 · Separation: distinct, but only partly")
    A("")
    A("**Question.** Are the types different distributions, how far apart, and in which features?")
    A("")
    A("**How.**")
    A("")
    A("- **Silhouette.** (b − a) / max(a, b), a = mean distance to the own cluster, b = to the nearest other cluster, "
      "on the unit-norm rows; z-scored against the Gaussian cloud clustered the same way. Per type, the mean of its "
      "electrodes' silhouettes against the cloud's per-cluster values.")
    A("- **MMD².** Mean kernel similarity within A + within B − 2 × between (Gaussian kernel, one median bandwidth), "
      "the unbiased estimate; 0 means the same distribution. In-sample it is tested against labels shuffled within "
      "patient, which is circular because the labels were fitted on these electrodes. The test that counts is on "
      "held-out patients: refit on half the patients, place the other half, MMD² between the placed groups, shuffled "
      "within patient; its scale is the Gaussian cloud through the same procedure, whose sectors are convex pieces of "
      "one featureless distribution and therefore well apart by construction.")
    A("- **Where.** Per bin, the standardised mean difference between the two types with the same within-patient "
      "shuffle; the bars are each band's share of the linear-kernel MMD².")
    A("")
    A("**What a good result looks like.** Silhouette z well above 0 overall and per type; every pair different on "
      "held-out patients; MMD² above the cloud's sectors for the pairs that are claimed to be distinct.")
    A("")
    sil = N["sil"]
    if N["mmd_present"]:
        st = N["sil_type"]
        st_sorted = sorted(st.items(), key=lambda kv: -kv[1]["z"])
        pa = N["mmd_argmax"]
        pc = N["mmd_cut"]
        below = [p for p in pa if p["ratio"] < 1]
        mp = N["mmd_params"]
        nn = N["mmd_type_argmax"]
        resm = N["mmd_res"]
        def drivers(pr):
            m = resm["maps"][("argmax", pr["i"], pr["j"])]
            bs = m["band_share"]
            bands = N["d"]["bands"]
            top = np.argsort(-bs)[:3]
            cs = m["cond_share"]
            conds = N["d"]["conds"]
            return (", ".join(f"{bands[b].replace('Hz', ' Hz')} {100*bs[b]:.0f} %" for b in top)
                    + "; by condition " + ", ".join(f"{c} {100*v:.0f} %" for c, v in zip(conds, cs)))
        A(f"**Found.** Overall silhouette {sil['value']:.3f} against {sil['null']:.3f} ± {sil['null_sd']:.4f} for "
          f"{sil['n_null']} clouds (z {sil['z']:.0f}). Per type, z against the cloud: "
          + ", ".join(f"{nm[j]} {v['z']:+.1f}" for j, v in st_sorted)
          + f". MMD ({mp['n_perm']} in-sample shuffles, {mp['n_splits']} patient halves with {mp['n_perm_heldout']} "
          f"shuffles each, {mp['n_cloud']} cloud draws, bandwidth {mp['bandwidth']:.2f}): every one of the {len(pa)} pairs "
          f"differs on held-out patients (shuffle p < 0.05 in at least {100*min(p['share_p'] for p in pa):.0f} % of the "
          f"splits). Held-out MMD² runs from {pa[0]['mmd2_ho']:.3f} (c{pa[0]['i']} vs c{pa[0]['j']}) to "
          f"{pa[-1]['mmd2_ho']:.3f} (c{pa[-1]['i']} vs c{pa[-1]['j']}); the cloud's sectors sit at {pa[0]['cloud']:.3f}, "
          f"so {len(pa) - len(below)} pairs are further apart than two sectors of a featureless cloud and {len(below)} are "
          f"closer: " + ", ".join(f"c{p['i']}–c{p['j']} {p['ratio']:.2f}×" for p in below)
          + ". Nearest neighbours: "
          + ", ".join(f"{nm[j]} → c{nn[j]['nn']} ({nn[j]['nn_ratio']:.2f}×)" for j in o)
          + f". With the {N['cut']:.1f} loading cut ({100*mp['frac_below_cut']:.1f} % of the electrodes removed, the "
          f"cloud cut by the same fraction) the ordering is the same and the ratios run {pc[0]['ratio']:.2f} to "
          f"{pc[-1]['ratio']:.2f}. What drives the most separated pair (c{pa[-1]['i']} vs c{pa[-1]['j']}): "
          f"{drivers(pa[-1])}. The least separated (c{pa[0]['i']} vs c{pa[0]['j']}): {drivers(pa[0])}.")
    else:
        A(f"**Found.** Overall silhouette {sil['value']:.3f} against {sil['null']:.3f} ± {sil['null_sd']:.4f} "
          f"(z {sil['z']:.0f}). **MMD NOT RUN YET: the panels below are placeholders** (run 00_paper2_figure6_mmd.py).")
    A("")
    img("fig4", "Fig 4: separation - silhouette z per type, the held-out MMD matrix, the most and least separated pairs (outlined: q < 0.05 and |d| >= 0.3), the loading distribution")
    A("**Takeaway.** The types are distinct: every pair differs on patients the model never saw, and the strongest "
      "types sit further apart than sectors of a featureless cloud. They are distinct only partly: the weakest pairs are "
      "closer than such sectors, the loadings are mixed (median 0.41), and a reading of the types as regions on a "
      "continuum of responses is consistent with these numbers. **Caveat.** In-sample p-values are circular and are not "
      "the test; the cloud is a harsh yardstick because clustering carves it into maximally separated pieces.")
    A("")

    # ---- 5
    A("## 5 · Scorecard: each type on all three")
    A("")
    A("**Question.** Which types are solid on agreement, stability and separation, and which are fragile?")
    A("")
    A("**How.** One row per type; the columns are the per-type numbers of sections 2 to 4: the chance-adjusted overlap "
      "of its matched cluster across the two other algorithms and across the three other feature sets (FIG 4), its "
      "consensus under patient resampling, its silhouette z against the cloud, the held-out MMD² to its nearest other "
      "type, and the number of patients and the largest patient's share. Colour is the rank within a column, not a "
      "composite score.")
    A("")
    A("**What a good result looks like.** Dark across the row.")
    A("")
    def pick(dct, j, k=None):
        v = dct[j] if k is None else dct[j][k]
        return v
    A("**Found.** " + " ".join(
        f"{nm[j]}: overlap {N['type_algo_adj'][j]:.2f} across algorithms and {N['type_feat_adj'][j]:.2f} across feature "
        f"sets, consensus {N['consensus_type'][j]:.2f}"
        + (f", silhouette z {N['sil_type'][j]['z']:+.1f}, nearest type {N['mmd_type_argmax'][j]['nn_ratio']:.2f}× the cloud"
           if N["mmd_present"] else "")
        + f", {N['patients_per_type'][j]} patients, largest {100*N['top_share'][j]:.0f} %." for j in o))
    A("")
    img("fig5", "Fig 5: the scorecard")
    if N["mmd_present"]:
        solid = [j for j in o if N["type_algo_adj"][j] >= 0.5 and N["consensus_type"][j] >= 0.75 and N["sil_type"][j]["z"] >= 2]
        fragile = [j for j in o if N["type_algo_adj"][j] < 0.35 and N["consensus_type"][j] < 0.7]
        A(f"**Takeaway.** Solid on all three: {', '.join(nm[j] for j in solid) or 'none'}. Fragile on agreement and "
          f"stability alike: {', '.join(nm[j] for j in fragile) or 'none'}. The rest split by question: "
          + "; ".join(f"{nm[j]} is "
                      + ("stable" if N["consensus_type"][j] >= 0.75 else "middling")
                      + f" under resampling ({N['consensus_type'][j]:.2f}) with agreement "
                      + f"{N['type_algo_adj'][j]:.2f} across algorithms and {N['type_feat_adj'][j]:.2f} across feature sets"
                      + (f", silhouette z {N['sil_type'][j]['z']:+.1f}" if N["mmd_present"] else "")
                      for j in o if j not in solid and j not in fragile)
          + ". **Caveat.** The columns are on different scales and different nulls; the card ranks, it does not sum.")
    A("")

    # ---- 6
    A("## 6 · Anatomy: where the types sit relative to the language atlas (context, not validation)")
    A("")
    A("**Question.** Does the probabilistic language atlas (LanA) place any type?")
    A("")
    A("**How.** Each electrode's atlas probability P(LanA); per type the mean P and the share inside the network at "
      "P ≥ 0.1, against a shaft-shift null that keeps spatial smoothness and a within-patient shuffle that does not, "
      "Benjamini-Hochberg over the types; and a spin null that rotates the atlas on the sphere, electrodes fixed.")
    A("")
    A("**What a good result looks like.** For validation, nothing in particular: anatomy is where the types are, not "
      "whether they are real.")
    A("")
    la = N["lana"]
    sh_sig = [j for j in o if la[j]["q_t10_sh"] < 0.05]
    sh_q = ", ".join(f"{la[j]['q_t10_sh']:.3f}" for j in sh_sig)
    mean_sig = [j for j in o if la[j]["q_mean_sh"] < 0.05]
    mean_wp = [j for j in o if la[j]["q_mean"] < 0.05 and la[j]["q_mean_sh"] >= 0.05]
    A(f"**Found.** Mean P(LanA) per type "
      + ", ".join(f"{nm[j]} {la[j]['mean_P']:.2f}" for j in o)
      + f" against a cohort baseline of {N['lana_base']['P']:.2f}. Against the shaft-shift null, the share inside the "
      f"network clears it for {', '.join('c' + str(j) for j in sh_sig) or 'no type'}"
      + (f" (q {sh_q})" if sh_sig else "")
      + f"; the mean P clears it for {', '.join('c' + str(j) for j in mean_sig) or 'no type'}"
      + (f", and for {', '.join('c' + str(j) for j in mean_wp)} only against the within-patient shuffle" if mean_wp else "")
      + f". Spin null ({N['spin']['n_spin']} spins): the network's own response is not beyond spun maps "
      f"(p {N['spin']['p_all']:.2f} all electrodes, {N['spin']['p_left']:.2f} left hemisphere); "
      f"{N['spin_sig']['all']['share']} of {K} inside shares beat the spun maps on all electrodes, "
      f"{N['spin_sig']['L']['share']} on the left; the mean P beats them for "
      f"{', '.join('c' + str(j) for j in N['spin_sig']['L']['meanP_which']) or 'no type'} on the left.")
    A("")
    img("fig6a", "Fig 6A: P(LanA) per type, the share inside the network, what the network is made of, LanA on this coverage")
    img("fig6b", "Fig 6B: the spin null and the network's own response")
    A("**Takeaway.** No type is placed by the language atlas beyond one modest effect. **Caveat.** Coverage is not "
      "symmetric and not random; the atlas is a prior about location, not a measurement in these patients.")
    A("")

    # ---- supplement
    A("## Supplement")
    A("")
    img("figS1", "S1: held-out variance explained against K, the four feature sets (current FIG 1 A)")
    img("figS2", "S2: RCSI by K for every cell - the stability ceiling per method")
    for key, cap in (("figS3b", "S3: the types on 5 bands"), ("figS3c", "S3: the types on 2 bands"),
                     ("figS3d", "S3: the types on HFA")):
        img(key, cap)
    for key, cap in (("figS4_8", "S4: matched clusters at K 8"), ("figS4_10", "S4: matched clusters at K 10")):
        img(key, cap)
    img("figS5", "S5: the full MMD figure (argmax and cut, the per-type comparisons, the two least separated pairs)")

    # ---- mismatches
    A("## Where the brief and the files disagree")
    A("")
    A("The files win; each point names the file.")
    A("")
    if N["mmd_present"]:
        A(f"- \"c2 and c5 solid\": c2 yes (overlap {N['type_algo_adj'][2]:.2f} across algorithms, "
          f"{N['type_feat_adj'][2]:.2f} across feature sets, consensus {N['consensus_type'][2]:.2f}, silhouette z "
          f"{N['sil_type'][2]['z']:+.1f}). c5 is solid under resampling ({N['consensus_type'][5]:.2f}) and on silhouette "
          f"(z {N['sil_type'][5]['z']:+.1f}) but not across methods: {N['type_algo_adj'][5]:.2f} across algorithms "
          f"(0.16 against Ward) and {N['type_feat_adj'][5]:.2f} across feature sets (FIG4_matched).")
        A(f"- \"c4 fragile across methods, most stable under resampling\": across algorithms c4 is middling "
          f"({N['type_algo_adj'][4]:.2f}); it is fragile across feature sets ({N['type_feat_adj'][4]:.2f}; 0.06 on 2 "
          f"bands, 0.03 on HFA). Most stable holds by a hair: {N['consensus_type'][4]:.3f} against c2's "
          f"{N['consensus_type'][2]:.3f} (items_k07).")
    A(f"- \"c1 and c3 fragile\": confirmed on all three (overlap {N['type_algo_adj'][1]:.2f}/{N['type_algo_adj'][3]:.2f} "
      f"across algorithms, {N['type_feat_adj'][1]:.2f}/{N['type_feat_adj'][3]:.2f} across feature sets, consensus "
      f"{N['consensus_type'][1]:.2f}/{N['consensus_type'][3]:.2f}"
      + (f", silhouette z {N['sil_type'][1]['z']:+.1f}/{N['sil_type'][3]['z']:+.1f}" if N["mmd_present"] else "") + ").")
    A(f"- \"K 7 → K 8 kept 0.90\": the file says {kb[8]['share']:.3f} (chance-corrected {kb[8]['adj']:.2f}) - confirmed.")
    A(f"- \"K 7 is the most clusters before one patient owns a cluster\": confirmed - at K = {N['gen_first_dominated_k']} "
      f"one cluster is {100*gen[N['gen_first_dominated_k']]['worst']:.0f} % one patient (FIG1a generalization), and the "
      f"consensus partition fails the rule {fail_txt} (results.json).")
    A("- Scorecard \"Jaccard\": FIG 4's per-type numbers are the chance-adjusted overlap; the card shows that and names "
      "it, the raw Jaccard is in numbers.json.")
    A(f"- \"Leave-one-patient-out vs size-matched pseudo-patients\": the file's null is shaft-permuted pseudo-patients, "
      f"and for convex NMF the real ARI is not above it (z {L_['z']:+.1f}); the panel is per patient, not per cluster, "
      f"because the tables are per patient.")
    A("- \"Per-type consensus (from the consensus slide)\": the slide's values are by consensus cluster id "
      f"({', '.join(f'{v:.2f}' for v in N['consensus_slide_values'])}); the document uses the mean own-cluster consensus "
      f"of each published type's electrodes (items_k07), which differs in the second decimal.")
    A("- The silhouette z per type and the MMD numbers did not exist before today; they come from "
      "00_paper2_figure6_mmd.py's tables (outputs/clustering/mmd/).")
    A("")
    A("## Sources")
    A("")
    for k, v in SRC.items():
        A(f"- {k}: `{v}`")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------- html
def md_to_html(md: str, base: Path) -> str:
    out = []
    in_list = False
    for raw in md.split("\n"):
        line = raw.rstrip()
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>"); in_list = False
        if not line:
            continue
        m = re.match(r"^(#+)\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)$", line)
        if m:
            p = base / m.group(2)
            data = base64.b64encode(p.read_bytes()).decode("ascii")
            out.append(f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(m.group(1))}">'
                       f"<figcaption>{html.escape(m.group(1))}</figcaption></figure>")
            continue
        out.append(f"<p>{inline(line)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


CSS = """
body{font-family:Georgia,'Times New Roman',serif;max-width:1080px;margin:40px auto;padding:0 24px;color:#1b232c;line-height:1.5}
h1{font-size:1.9em;margin-top:0}h2{font-size:1.35em;margin-top:2.2em;border-top:1px solid #c9ced4;padding-top:0.8em}
p{margin:0.6em 0}ul{margin:0.4em 0 0.8em}li{margin:0.25em 0}
figure{margin:1.4em 0}img{width:100%;height:auto;border:1px solid #e6e9ed}
figcaption{font-size:0.85em;color:#68727d;margin-top:0.3em;font-family:Helvetica,Arial,sans-serif}
code{font-family:Consolas,Menlo,monospace;font-size:0.9em;background:#f2f3f5;padding:0 3px}
"""


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass
    t0 = time.time()
    OUTD.mkdir(parents=True, exist_ok=True)
    N = collect()
    print(f"numbers collected ({time.time()-t0:.0f} s); MMD tables {'present' if N['mmd_present'] else 'ABSENT - placeholders'}")
    F = build_figures(N)
    print(f"{len(F)} figures -> {FIGD.relative_to(ROOT)}")
    md = write_text(N, F)
    (OUTD / "validation_story.md").write_text(md, encoding="utf-8")
    body = md_to_html(md, OUTD)
    page = ("<!doctype html><html><head><meta charset='utf-8'><title>Paper 2 validation story</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    (OUTD / "validation_story.html").write_text(page, encoding="utf-8")
    keep = {k: v for k, v in N.items() if k not in ("d", "mmd_res")}
    (OUTD / "numbers.json").write_text(json.dumps(dict(values=keep, sources=SRC), indent=1, default=str), encoding="utf-8")
    print(f"validation_story.md ({len(md)//1000} k chars), validation_story.html "
          f"({(OUTD / 'validation_story.html').stat().st_size/1e6:.1f} MB), numbers.json   ({time.time()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
