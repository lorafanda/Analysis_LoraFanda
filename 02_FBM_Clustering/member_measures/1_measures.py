"""Step 1 of member_measures (Lora, 2026-11-02): one row per electrode with every per-electrode
measure, for each feature set at its published K (convex NMF, v17, argmax labels).

RUN ORDER (Python 3.11, PYTHONIOENCODING=utf-8). Steps marked * take --fset 15band|5band|2band|hfa
(default: all four). Everything is written under member_measures/out/.
    1_measures.py *        the eight measures + P(LanA)            -> out/<fset>/measures.csv
    2_white_matter.py      % white matter around each contact       -> out/white_matter.csv
    3_spatial_grouping.py * same-type share of 10 other-patient nbrs -> out/<fset>/spatial_grouping.csv
    4_neurosynth.py        z on every NeuroSynth map + statistics    -> out/neurosynth_z.csv, neurosynth_maps.csv
    5_bayes_ranks.py *     Bayesian ranking of the types             -> out/<fset>/bayes_ranks.csv
    6_story_figures.py *   one figure per colouring per feature set  -> out/figures/<fset>_story_<colour>.png
    7_per_cluster.py       per 15-band type, the four feature sets   -> out/figures_per_cluster/c<j>_story_<colour>.png
    8_neurosynth_enrichment.py  clusters x NeuroSynth maps, and the
                           feature sets compared                      -> out/neurosynth_enrichment/*.png, *.csv
    9_neurosynth_bars.py   % of each cluster inside each map, as bars,
                           per feature set and per 15-band type       -> out/neurosynth_bars/*.png, *.csv
    10_neurosynth_labels.py  a NeuroSynth label per cluster (bubble
                           matrix, 8 maps; X = >50% one patient)      -> out/neurosynth_labels/*.png, *.csv
Every step except 2 and 4 takes --k published (default, out/) or --k 7 (all seven feature sets at
K = 7, out_k7/): see mm_common.py.

Measures: loading (own-type normalised loading), silhouette (unit-norm space of that feature set,
published labels), consensus (mean co-assignment with own type, consensus sweep M_k<K>), lopo / halves
(held-out hit rates, FIG 6's confusion_*_electrodes.csv), mmd_witness (mean Gaussian-kernel
similarity to own type minus to the nearest other, FIG 6's bandwidth for that cell), r_type_mean (r
with the leave-one-out type mean), gate_margin (best condition, the cache's gate table), P_lana.
Reads only."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples

import mm_common as C

F3 = C.load_module(C.SHARE / "00_paper2_figure3_lana.py", "mm_f3")
P2 = F3.P2
F0 = C.load_module(C.SHARE / "00_paper2_figure0_coverage.py", "mm_f0")
gp, d0, t0, g0, u0 = F0.prepare("concat_bands15", 7)          # the gate table is cohort-wide
gate = u0.set_index("key").best

for tag in C.tags_from_args():
    fset, K = C.FSETS[tag]
    print(f"\n=== {tag}: {fset} K = {K}")
    d = F3.load(fset, K)
    lab, X = np.asarray(d["lab"]), np.asarray(d["X"], float)
    keys = [f"{p}|{P2.norm(e)}" for p, e in zip(d["meta"].patient_id, d["meta"].electrode)]
    n = len(lab)
    M_ = {"loading": d["Gn"][np.arange(n), lab]}
    A = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    M_["silhouette"] = silhouette_samples(A, lab)

    cdir = C.OUTC / "consensus" / "concat_source_v17" / fset / "cnmf"
    fM, fL = cdir / f"M_k{K:02d}.npy", cdir / f"consensus_labels_k{K:02d}.csv"
    if not fL.exists():
        # the row order of every M_k*.npy is the cell's electrode order, the same at every K;
        # consensus_stability writes it only beside the K it summarised
        fL = next(iter(sorted(cdir.glob("consensus_labels_k*.csv"))), fL)
    cons = np.full(n, np.nan)
    if fM.exists() and fL.exists():
        Mc = np.load(fM)
        order = pd.read_csv(fL)
        ck = [f"{p}|{P2.norm(e)}" for p, e in zip(order.patient_id, order.electrode)]
        ix = pd.Series(range(len(ck)), index=ck).reindex(keys).to_numpy()
        if np.isnan(ix.astype(float)).any():
            raise SystemExit(f"{tag}: the consensus order does not cover the run")
        Mc = Mc[np.ix_(ix.astype(int), ix.astype(int))]
        for i in range(n):
            same = lab == lab[i]
            same[i] = False
            cons[i] = Mc[i, same].mean()
    else:
        print(f"  NOTE {tag}: no consensus sweep at K = {K} ({fM.name}) - consensus left blank")
    M_["consensus"] = cons

    mdir = C.OUTC / "mmd" / "concat_source_v17" / fset / "cnmf"
    man = json.loads((mdir / "manifest.json").read_text()) if (mdir / "manifest.json").exists() else {}
    for h in ("lopo", "halves"):
        f_ = mdir / f"confusion_{h}_electrodes.csv"
        if f_.exists() and int(man.get("k", -1)) == K:
            t = pd.read_csv(f_)
            t["key"] = [f"{p}|{P2.norm(e)}" for p, e in zip(t.patient, t.electrode)]
            M_[h] = t.set_index("key").hit_rate.reindex(keys).to_numpy(float)
        else:
            M_[h] = np.full(n, np.nan)
            print(f"  NOTE {tag}: FIG 6 has no held-out table at K = {K} - {h} left blank")

    if "bandwidth" in man:                             # FIG 6's own value for this feature set
        sigma = float(man["bandwidth"])
    else:                                              # FIG 6's recipe: median pairwise distance
        rs = np.random.default_rng(0)
        B = A[rs.choice(len(A), min(len(A), 2000), replace=False)]
        sqb = (B * B).sum(1)
        D2b = np.maximum(sqb[:, None] + sqb[None, :] - 2.0 * B @ B.T, 0.0)
        sigma = float(np.sqrt(np.median(D2b[np.triu_indices(len(B), 1)])))
    sq = (A * A).sum(1)
    Kmat = np.exp(-np.maximum(sq[:, None] + sq[None, :] - 2.0 * A @ A.T, 0.0) / (2.0 * sigma * sigma))
    np.fill_diagonal(Kmat, np.nan)
    mean_to = np.stack([np.nanmean(Kmat[:, lab == j], axis=1) for j in range(K)], axis=1)
    own_k = mean_to[np.arange(n), lab]
    mean_to[np.arange(n), lab] = -np.inf
    M_["mmd_witness"] = own_k - mean_to.max(1)

    sums = np.stack([X[lab == j].sum(0) for j in range(K)])
    cnt = np.bincount(lab, minlength=K)
    M_["r_type_mean"] = np.array([np.corrcoef(X[i], (sums[lab[i]] - X[i]) / (cnt[lab[i]] - 1))[0, 1]
                                  for i in range(n)])
    M_["gate_margin"] = gate.reindex(keys).to_numpy(float)

    out = pd.DataFrame(dict(key=keys, type=lab, P_lana=np.asarray(d["P_lana"], float), **M_))
    out.to_csv(C.fset_dir(tag) / "measures.csv", index=False)
    print(out.drop(columns="key").groupby("type").median().round(3).to_string())
    print("wrote", C.OUT / tag / "measures.csv")
