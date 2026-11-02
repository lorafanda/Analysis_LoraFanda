#!/usr/bin/env python3
"""compare_feature_variants.py - how different is the 15-band partition once the bands are
equalised? The newest convex NMF run of each VARIANT feature set is compared with the
published concat_bands15 partition at K 7, on the same electrodes (2026-11-02).

    concat_bands15z   the 15 bands, each z-scored across the cohort
    concat_bands15s   the 15 bands as per-band shape within each electrode

    python compare_feature_variants.py                          # both variants
    python compare_feature_variants.py --variants concat_bands15z --k 8

Per variant: the one-patient rule over the swept K (the chosen K is the largest with no
cluster over half one patient), ARI and NMI against the published K 7 labels, the
Hungarian match of its clusters to the published types on shared electrodes with the
Jaccard of each pair and the share of each type's electrodes that stays with its match,
where the Figure 1 exemplars land, the share of the feature vector each band group
carries, and the variant clusters drawn as the dB means of their members beside the
matched published types. Reads outputs only; writes
outputs/clustering/feature_variants/<cache>/ (summary.md, summary.json, PNGs).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parent
_sp = importlib.util.spec_from_file_location("p2", ROOT / "00_Paper2_Figures.py")
P2 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(P2)
sys.path.insert(0, str(ROOT / "functions"))
import lf_runs as LR                                      # noqa: E402

REF, REF_K = "concat_bands15", 7
VARIANTS = {"concat_bands15z": "15 bands, each z-scored across the cohort",
            "concat_bands15s": "15 bands, per-band shape within each electrode"}
GROUPS = {"1-13 Hz": [0, 1, 2], "13-30 Hz": [3, 4], "30-70 Hz": [5, 6],
          "70-170 Hz": [7, 8, 9], "170-400 Hz": [10, 11, 12, 13, 14]}
EXEMPLARS = [("EL033", "PHG_R12", "Figure 1 production row"),
             ("PAT_1327", "CPG13", "Figure 1 beta-decrease row"),
             ("PAT_3965", "had6", "Figure 1 broadband-decrease row"),
             ("PAT_5533", "IPD3", "Figure 1 audio row"),
             ("EL043", "IOG7", "Figure 1 visual row"),
             ("EL042", "STG_R5", "most representative c1"),
             ("EL035", "TTG_R1", "types-figure example")]
VLIM = 2.0
NL = chr(10)


def keys_of(meta):
    return [f"{p}|{P2.norm(e)}" for p, e in zip(meta["patient_id"], meta["electrode"])]


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def band_share(X):
    """Median over electrodes of the share of the unit-normed vector each band group holds."""
    U = unit(np.asarray(X, float)).reshape(len(X), 15, -1)
    E = (U ** 2).sum(2)
    return {g: float(np.median(E[:, idx].sum(1))) for g, idx in GROUPS.items()}


def one_patient_rule(labels_by_k, patients):
    rows = []
    for col in labels_by_k.columns:
        if not col.startswith("k_"):
            continue
        k = int(col[2:]); lab = labels_by_k[col].to_numpy(int)
        worst, ndom = 0.0, 0
        for c in range(k):
            m = lab == c
            if not m.any():
                continue
            share = float(pd.Series(patients[m]).value_counts(normalize=True).iloc[0])
            worst = max(worst, share); ndom += int(share > 0.5)
        rows.append(dict(k=k, worst_share=worst, n_dominated=ndom))
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def match(lab_v, kv, lab_r, kr):
    """Hungarian assignment of variant clusters to published types on shared electrodes."""
    M = np.zeros((kv, kr), int)
    for a, b in zip(lab_v, lab_r):
        M[a, b] += 1
    ri, ci = linear_sum_assignment(-M)
    sv = np.bincount(lab_v, minlength=kv); sr = np.bincount(lab_r, minlength=kr)
    J = M / np.maximum(sv[:, None] + sr[None, :] - M, 1)
    return M, J, dict(zip(ri.tolist(), ci.tolist()))


def dB_means(C, lab, k):
    return [C[lab == c].mean(0) if (lab == c).any() else np.zeros(C.shape[1:]) for c in range(k)]


def analyse(v, ref, Cref, ref_order, pats, k_force, out):
    run = LR.newest_run("cnmf", v)
    meta = pd.read_csv(run / "labels.csv")
    kv = keys_of(meta)
    if kv != keys_of(ref["meta"]):
        raise SystemExit(f"{v}: {run.name} is not on the published run's electrodes "
                         f"({len(kv)} vs {len(ref['meta'])}); refit 240 and 242 on the current cache")
    Xv = np.load(run / "X_train.npy").astype(float)
    lbk = pd.read_csv(run / "cluster_labels_by_k.csv")
    rule = one_patient_rule(lbk, pats)
    clean = rule[rule.n_dominated == 0]
    k_rule = int(clean.k.max()) if len(clean) else None
    ks = sorted({REF_K} | ({k_force} if k_force else set()) | ({k_rule} if k_rule else set()))
    names = P2.TYPE_NAMES[(REF, REF_K)]
    res = dict(variant=v, label=VARIANTS.get(v, v), run=run.name, k_rule=k_rule,
               ks_swept=[int(k) for k in rule.k], rule=rule.to_dict("records"), at={},
               band_share=band_share(Xv))
    for k in ks:
        gf = run / "loadings_by_k" / f"G_k{k:02d}.npy"
        if not gf.exists():
            print(f"  {v}: no loadings at K {k} (sweep did not include it), skipped")
            continue
        G = np.load(gf).astype(float)
        Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
        lab = Gn.argmax(1)
        M, J, mp = match(lab, k, ref["lab"], REF_K)
        pairs = []
        for a, b in sorted(mp.items(), key=lambda ab: ref_order.index(ab[1])):
            pairs.append(dict(variant_cluster=int(a), n=int((lab == a).sum()),
                              published=f"c{b} {names.get(b, '')}".strip(),
                              jaccard=float(J[a, b]),
                              share_of_type_kept=float(M[a, b] / max((ref["lab"] == b).sum(), 1)),
                              share_of_cluster_from_type=float(M[a, b] / max((lab == a).sum(), 1))))
        unmatched = [dict(variant_cluster=int(a), n=int((lab == a).sum()),
                          closest=f"c{int(J[a].argmax())} {names.get(int(J[a].argmax()), '')}".strip(),
                          jaccard=float(J[a].max())) for a in range(k) if a not in mp]
        ex = []
        for p, e, why in EXEMPLARS:
            hit = [n for n, kk in enumerate(kv) if kk == f"{p}|{P2.norm(e)}"]
            if not hit:
                continue
            i = hit[0]; a = int(lab[i]); b = mp.get(a); rb = int(ref["lab"][i])
            ex.append(dict(electrode=f"{p} {e}", why=why,
                           published=f"c{rb} {names.get(rb, '')}".strip(),
                           variant_cluster=a,
                           matched=(f"c{b} {names.get(b, '')}".strip() if b is not None else "unmatched"),
                           own_loading=float(Gn[i].max()),
                           loading_on_matched_audio=float(Gn[i, [a_ for a_, b_ in mp.items() if b_ == 1][0]]) if any(b_ == 1 for b_ in mp.values()) else None))
        res["at"][int(k)] = dict(ari=float(adjusted_rand_score(ref["lab"], lab)),
                                 nmi=float(normalized_mutual_info_score(ref["lab"], lab)),
                                 stay=float(np.mean([mp.get(a) == b for a, b in zip(lab, ref["lab"])])),
                                 median_own_loading=float(np.median(Gn.max(1))),
                                 pairs=pairs, unmatched=unmatched, exemplars=ex)
        fig_centroids(v, k, lab, mp, J, ref, Cref, ref_order, out)
        fig_matching(v, k, J, mp, ref_order, names, out)
    return res


def _cube_img(ax, cu, title, nt):
    """One (cond, band, t) cube as a 15 x 90 dB image, conditions side by side."""
    im = ax.imshow(np.concatenate(list(cu), axis=1), aspect="auto", origin="lower",
                   cmap="bwr", vmin=-VLIM, vmax=VLIM)
    for x in (nt, 2 * nt):
        ax.axvline(x - 0.5, color="k", lw=0.6)
    for x in (nt / 2, 1.5 * nt, 2.5 * nt):
        ax.axvline(x - 0.5, color="k", lw=0.3, ls=":")
    ax.set_title(title, fontsize=7.5, loc="left")
    ax.set_xticks([nt / 2, 1.5 * nt, 2.5 * nt]); ax.set_xticklabels(["audio", "picture", "sentences"], fontsize=6.5)
    ax.tick_params(length=0)
    return im


def fig_centroids(v, k, lab, mp, J, ref, Cref, ref_order, out):
    """Top row: the published type each variant cluster matched (dB mean of its members).
    Bottom row: the variant cluster, also as the dB mean of ITS members (so both rows are
    in dB whatever units the variant was fitted in). Unmatched clusters come last."""
    names = P2.TYPE_NAMES[(REF, REF_K)]
    rm = dB_means(Cref, ref["lab"], REF_K); vm = dB_means(Cref, lab, k)
    cols = sorted(mp.items(), key=lambda ab: ref_order.index(ab[1]))
    cols = cols + [(a, None) for a in range(k) if a not in mp]
    n = len(cols)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n + 1.2, 4.6), dpi=150, squeeze=False)
    for j, (a, b) in enumerate(cols):
        if b is not None:
            _cube_img(axes[0, j], rm[b], f"published c{b} {names.get(b, '')}" + NL + f"n {int((ref['lab'] == b).sum())}", ref["nt"])
            ttl = f"variant cluster {a}   n {int((lab == a).sum())}" + NL + f"Jaccard {J[a, b]:.2f} · keeps {100 * J[a, b] * 0 + 100 * (np.sum((lab == a) & (ref['lab'] == b)) / max((ref['lab'] == b).sum(), 1)):.0f} % of the type"
        else:
            axes[0, j].axis("off"); axes[0, j].set_title("no published partner", fontsize=7.5, loc="left")
            ttl = f"variant cluster {a}   n {int((lab == a).sum())}" + NL + f"closest c{int(J[a].argmax())} (Jaccard {J[a].max():.2f})"
        im = _cube_img(axes[1, j], vm[a], ttl, ref["nt"])
        for r in (0, 1):
            axes[r, j].set_yticks([0, 2, 4, 6, 9, 14]); axes[r, j].set_yticklabels([ref["bands"][i] for i in (0, 2, 4, 6, 9, 14)] if j == 0 else [], fontsize=6)
    fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01, label="dB vs baseline (members' mean)")
    fig.suptitle(f"{v} at K {k}  vs  the published {REF} types (K {REF_K}): matched pairs in the paper's order", fontsize=9)
    fig.savefig(out / f"{v}_centroids_K{k}.png", facecolor="white", bbox_inches="tight"); plt.close(fig)


def fig_matching(v, k, J, mp, ref_order, names, out):
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * k + 1.6), dpi=150)
    order_v = [a for a, _ in sorted(mp.items(), key=lambda ab: ref_order.index(ab[1]))] + [a for a in range(k) if a not in mp]
    Jo = J[np.ix_(order_v, ref_order)]
    ax.imshow(Jo, cmap="Greys", vmin=0, vmax=1)
    for i in range(Jo.shape[0]):
        for j in range(Jo.shape[1]):
            ax.text(j, i, f"{Jo[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if Jo[i, j] > 0.55 else P2.INK,
                    fontweight="bold" if mp.get(order_v[i]) == ref_order[j] else "normal")
    ax.set_xticks(range(REF_K)); ax.set_xticklabels([f"c{b}" + NL + names.get(b, "") for b in ref_order], fontsize=6.5)
    ax.set_yticks(range(k)); ax.set_yticklabels([f"variant {a}" for a in order_v], fontsize=7)
    ax.set_xlabel("published type", fontsize=8); ax.tick_params(length=0)
    ax.set_title(f"{v} K {k}: Jaccard of shared electrodes (bold = Hungarian match)", fontsize=8.5, loc="left")
    fig.savefig(out / f"{v}_matching_K{k}.png", facecolor="white", bbox_inches="tight"); plt.close(fig)


def fig_rule(rules, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=150)
    for name, r in rules.items():
        ax.plot(r.k, 100 * r.worst_share, marker="o", ms=3.5, lw=1.4, label=name)
    ax.axhline(50, color=P2.RED if hasattr(P2, "RED") else "red", lw=0.8, ls="--")
    ax.set_xlabel("K"); ax.set_ylabel("largest share of a cluster held by one patient (%)")
    ax.set_ylim(0, 102); ax.legend(fontsize=7, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("the one-patient rule: the chosen K is the last before a cluster is over half one patient", fontsize=8, loc="left")
    fig.savefig(out / "one_patient_rule.png", facecolor="white", bbox_inches="tight"); plt.close(fig)


def fig_shares(shares, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=150)
    names = list(shares); x = np.arange(len(GROUPS)); w = 0.8 / len(names)
    for i, nm_ in enumerate(names):
        ax.bar(x + (i - (len(names) - 1) / 2) * w, [100 * shares[nm_][g] for g in GROUPS], width=w, label=nm_)
    ax.set_xticks(x); ax.set_xticklabels(list(GROUPS), fontsize=8)
    ax.set_ylabel("share of the unit-normed vector (%)"); ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("what the cosine distance weighs: median electrode, per band group", fontsize=8.5, loc="left")
    fig.savefig(out / "band_shares.png", facecolor="white", bbox_inches="tight"); plt.close(fig)


def write_summary(results, ref_rule, ref_share, out, cache):
    L = []; A = L.append
    A(f"# Band-equalised variants of {REF} against the published K {REF_K} partition")
    A("")
    A(f"*Built {time.strftime('%Y-%m-%d %H:%M')} on cache {cache}. Reads the newest convex NMF run of each variant; nothing published is touched.*")
    A("")
    A("## The one-patient rule")
    A("")
    A("| K | " + " | ".join([REF] + [r["variant"] for r in results]) + " |")
    A("|---|" + "---|" * (1 + len(results)))
    ks = sorted(set(ref_rule.k) | set(k for r in results for k in r["ks_swept"]))
    for k in ks:
        cells = []
        for r in [dict(rule=ref_rule.to_dict("records"))] + results:
            row = [x for x in r["rule"] if x["k"] == k]
            cells.append(f"{100 * row[0]['worst_share']:.0f} %" + (" *" if row and row[0]["n_dominated"] else "") if row else "")
        A(f"| {k} | " + " | ".join(cells) + " |")
    A("")
    A("Largest share of a cluster held by one patient; * = a cluster over half one patient. Chosen K by the rule: "
      + ", ".join([f"{REF} {int(ref_rule[ref_rule.n_dominated == 0].k.max())}"] + [f"{r['variant']} {r['k_rule']}" for r in results]) + ".")
    A("")
    A("## What the distance weighs")
    A("")
    A("| band group | " + " | ".join([REF] + [r["variant"] for r in results]) + " |")
    A("|---|" + "---|" * (1 + len(results)))
    for g in GROUPS:
        A(f"| {g} | " + " | ".join(f"{100 * s[g]:.0f} %" for s in [ref_share] + [r["band_share"] for r in results]) + " |")
    A("")
    for r in results:
        A(f"## {r['variant']} · {r['label']} · run {r['run']}")
        A("")
        for k, at in r["at"].items():
            A(f"### at K {k}" + (" (the paper's K)" if k == REF_K else "") + (" (its own K by the rule)" if k == r["k_rule"] else ""))
            A("")
            A(f"ARI {at['ari']:.2f}, NMI {at['nmi']:.2f} against the published labels; {100 * at['stay']:.0f} % of the electrodes stay with the type their cluster is matched to; median own loading {at['median_own_loading']:.2f}.")
            A("")
            A("| variant cluster | n | matched published type | Jaccard | keeps of the type | of the cluster from that type |")
            A("|---|---|---|---|---|---|")
            for p in at["pairs"]:
                A(f"| {p['variant_cluster']} | {p['n']} | {p['published']} | {p['jaccard']:.2f} | {100 * p['share_of_type_kept']:.0f} % | {100 * p['share_of_cluster_from_type']:.0f} % |")
            for u in at["unmatched"]:
                A(f"| {u['variant_cluster']} | {u['n']} | (no partner; closest {u['closest']}) | {u['jaccard']:.2f} | | |")
            A("")
            A("| electrode | why it is here | published | variant cluster | matched type | own loading |")
            A("|---|---|---|---|---|---|")
            for e in at["exemplars"]:
                A(f"| {e['electrode']} | {e['why']} | {e['published']} | {e['variant_cluster']} | {e['matched']} | {e['own_loading']:.2f} |")
            A("")
            A(f"![{r['variant']} K {k} centroids]({r['variant']}_centroids_K{k}.png)")
            A("")
            A(f"![{r['variant']} K {k} matching]({r['variant']}_matching_K{k}.png)")
            A("")
    A("![one-patient rule](one_patient_rule.png)")
    A("")
    A("![band shares](band_shares.png)")
    A("")
    (out / "summary.md").write_text(NL.join(L), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(dict(cache=cache, ref=REF, ref_k=REF_K, ref_rule=ref_rule.to_dict("records"),
                                                      ref_band_share=ref_share, variants=results), indent=1, default=str), encoding="utf-8")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try: stream.reconfigure(errors="replace")
            except Exception: pass
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--k", type=int, default=None, help="also compare at this K (K 7 and each variant's rule K are always done)")
    a = ap.parse_args()
    t0 = time.time()
    cache = LR.current_cache()
    out = ROOT / "outputs" / "clustering" / "feature_variants" / cache
    out.mkdir(parents=True, exist_ok=True)
    ref = P2.load_run(REF, REF_K, "cnmf"); ref["fset"] = REF
    Cref = P2.cube(ref["X"], ref)
    ref_means = np.stack([Cref[ref["lab"] == j].mean(0) for j in range(REF_K)])
    ref_order = [int(j) for j in P2.block_order(ref_means, ref)[0]]
    pats = np.asarray(ref["patient"]).astype(str)
    ref_rule = one_patient_rule(pd.read_csv(ref["run"] / "cluster_labels_by_k.csv"), pats)
    ref_share = band_share(ref["X"])
    print(f"published {REF} run {ref['run'].name}: {len(pats)} electrodes, order {ref_order}")
    results, rules, shares = [], {REF: ref_rule}, {REF: ref_share}
    for v in a.variants:
        try:
            r = analyse(v, ref, Cref, ref_order, pats, a.k, out)
        except (FileNotFoundError, LR.StaleRunError) as e:
            print(f"{v}: no run yet ({e}); run 240 then 242 with --feature-set {v}")
            continue
        results.append(r); rules[v] = pd.DataFrame(r["rule"]); shares[v] = r["band_share"]
        for k, at in r["at"].items():
            print(f"{v} K {k}: ARI {at['ari']:.2f}  NMI {at['nmi']:.2f}  stay {100 * at['stay']:.0f} %  rule K {r['k_rule']}")
    if not results:
        return 1
    fig_rule(rules, out); fig_shares(shares, out)
    write_summary(results, ref_rule, ref_share, out, cache)
    print(f"written {out}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
