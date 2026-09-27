#!/usr/bin/env python3
"""
make_cnmf_three_roles_explainer.py - convex NMF at k = 4 on forty electrodes chosen to be
four things the eye can tell apart (auditory, visual, motor, and a "negative network" that
sits below baseline for the whole trial), then the arithmetic of the factorisation, one
matrix entry at a time.

    python make_cnmf_three_roles_explainer.py
    python make_cnmf_three_roles_explainer.py --per 10 --bins 10 --k 4

(The file keeps its name from the first, three-group version; Lora asked for the negative
network as a fourth group on 2026-09-27 so the negative values have something to contrast.)

E13 (make_cnmf_on_data_explainer.py) shows the method on six arbitrary electrodes at k = 2.
This one asks the harder question: if the sample is built from three response types, does
the factorisation find them, and what exactly is computed on the way? Every number is a
real measurement of the newest concat_source_v<N> cache: high-gamma (70-150 Hz) averaged
into `bins` time bins per condition, dB re baseline, signed, through the pipeline's own
lf_decompose.convex_nmf. The first half of each condition's bins is the stimulus, the
second half the response (config.proportions = (0, .5, .5)).

HOW THE THIRTY ARE CHOSEN - by rule, nothing by hand.
  Window means per electrode on the binned profile: sa = audio stimulus half, sp / sr =
  picture / reading stimulus half, resp = response half averaged over the three
  conditions, sv = (sp + sr) / 2, stim = (sa + sp + sr) / 3. Anatomy is the review
  bundle's aparc label (activity_viz/review/contacts.json).
    auditory  sa >= 1.5 dB and sv <= 0.5 dB on superior / transverse temporal contacts,
              ranked by sa - sv, at most 3 per patient
    visual    sv >= 0.7, sa <= 0.6, sp and sr >= 0.5, resp <= sv, ranked by sv - sa; no
              per-patient cap, because only two patients of the cohort carry a visual-only
              response at all (EL043's occipital shaft and PAT_3975)
    motor     resp >= 1.0 and resp - stim >= 1.0 on pre-/postcentral or pars opercularis
              contacts, ranked by resp - stim, at most 3 per patient
    negative  mean over the whole profile <= -0.8 dB and no bin above 0.5 dB (below
              baseline throughout, any anatomy), ranked by the mean, at most 3 per patient

WHAT IS COMPUTED, each with its numbers on the figure and in the JSON
  X (40 x 30)          electrodes x [audio bins | picture bins | reading bins]
  A = X X' (40 x 40)   split into A+ and A-: one positive and one negative entry worked
  W (40 x 4)           columns sum to 1;  F = W'X (4 x 30) are the components, in dB
  G (40 x 4)           loadings;  X^ = G F is the reconstruction
  one multiplicative update of one W entry, from the pipeline's own initialisation
  the fit ||X - G W'X|| / ||X||, and argmax(G) against the three groups

Outputs: explainers/E14_cnmf_three_roles.png and .json (make_cnmf_method_block.py reads
the JSON, so the words on the site cannot drift from the picture).
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import textwrap
from contextlib import redirect_stdout
from itertools import permutations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))
import lf_decompose as LD                                             # noqa: E402

OUT = ROOT / "outputs" / "clustering" / "explainers"
REVIEW = ROOT / "outputs" / "250_recon" / "fsaverage" / "activity_viz" / "review" / "contacts.json"
INK, MUTED, GREEN = "#1b232c", "#68727d", "#1b7837"
GROUPS = ("auditory", "visual", "motor", "negative")
GCOL = {"auditory": "#c1121f", "visual": "#2471a3", "motor": "#1b7837", "negative": "#5b2c83"}
WORD = {3: "three", 4: "four", 5: "five", 30: "thirty", 40: "forty", 50: "fifty"}
MONO = {"family": "DejaVu Sans Mono"}
HG = (70.0, 150.0)
FMAX = 398.4375                        # 102 x 3.90625 Hz, the last bin of the 0-400 cube
ANAT_AUD = re.compile(r"superior ?temporal|transverse ?temporal", re.I)
ANAT_MOT = re.compile(r"precentral|postcentral|pars ?opercularis", re.I)


# ---- data ----------------------------------------------------------------------------------
def newest_cache() -> Path:
    d = ROOT / "outputs" / "_dataset"
    caches = sorted(d.glob("concat_source_v*"), key=lambda p: int(p.name.rsplit("v", 1)[1]))
    if not caches:
        raise SystemExit("no concat_source_v<N> cache")
    return caches[-1]


def load_cohort(n_bins: int):
    """Every cohort electrode (three conditions, gate passed in at least one) as 3*n_bins numbers."""
    cache = newest_cache()
    params = json.load(open(cache / "params.json"))
    meta = pd.read_parquet(cache / "df_meta.parquet")
    freqs = np.linspace(0, FMAX, params["n_freq"])
    band = np.where((freqs >= HG[0]) & (freqs <= HG[1]))[0]
    lo, hi = int(band[0]), int(band[-1])
    X3 = np.load(cache / "X_3d.npy", mmap_mode="r")
    meta = meta.assign(key=meta.patient_id.astype(str) + "|" + meta.electrode.astype(str))
    per = meta.groupby("key").condition.nunique()
    hot = meta.groupby("key").high_activity.any()
    keys = sorted(set(per[per == 3].index) & set(hot[hot].index))
    conds = list(params["conditions"])
    idx = {(k, c): int(i) for k, c, i in zip(meta.key, meta.condition, meta.sample_idx)}
    rows = []
    for k in keys:
        parts = []
        for c in conds:
            cube = np.asarray(X3[idx[(k, c)], lo:hi + 1, :], dtype=np.float64)
            parts.append(np.nanmean(cube, 0).reshape(n_bins, -1).mean(1))
        rows.append(np.concatenate(parts))
    X = pd.DataFrame(np.vstack(rows), index=keys)
    return X, conds, cache.name, len(band), int(params["n_freq"])


def anatomy() -> dict:
    con = json.load(open(REVIEW, encoding="utf-8"))
    # Bern patients carry the fsaverage aparc label; HUG patients their own table's label
    # ('R postcentral · WM 100 %'), which is what the review page shows for them too
    return {c["patient"] + "|" + c["name"]: (c.get("aparc") or c.get("label") or "") for c in con}


def short_anat(s: str) -> str:
    """'R postcentral · WM 100 %' -> 'postcentral';  'lateraloccipital' -> 'lateral occipital'."""
    s = s.split("·")[0].strip()
    s = re.sub(r"^[LR]\s+", "", s)
    for a, b in (("lateraloccipital", "lateral occipital"), ("superiortemporal", "superior temporal"),
                 ("transversetemporal", "transverse temporal"), ("inferiortemporal", "inferior temporal"),
                 ("parsopercularis", "pars opercularis")):
        s = s.replace(a, b)
    return s


def windows(X: pd.DataFrame, nb: int) -> pd.DataFrame:
    h = nb // 2
    a, p, r = X.iloc[:, 0:nb], X.iloc[:, nb:2 * nb], X.iloc[:, 2 * nb:3 * nb]
    S = pd.DataFrame(dict(sa=a.iloc[:, :h].mean(1), ra=a.iloc[:, h:].mean(1),
                          sp=p.iloc[:, :h].mean(1), rp=p.iloc[:, h:].mean(1),
                          sr=r.iloc[:, :h].mean(1), rr=r.iloc[:, h:].mean(1)))
    S["sv"] = (S.sp + S.sr) / 2
    S["resp"] = (S.ra + S.rp + S.rr) / 3
    S["stim"] = (S.sa + S.sp + S.sr) / 3
    S["mean"] = X.mean(1)
    S["mx"] = X.max(1)
    return S


def select(S: pd.DataFrame, anat: dict, per: int):
    S = S.assign(pat=[k.split("|")[0] for k in S.index], anat=[anat.get(k, "") for k in S.index])
    aud = S[(S.sa >= 1.5) & (S.sv <= 0.5) & S.anat.str.contains(ANAT_AUD)].assign(score=lambda d: d.sa - d.sv)
    vis = S[(S.sv >= 0.7) & (S.sa <= 0.6) & (S.sp >= 0.5) & (S.sr >= 0.5) & (S.resp <= S.sv)].assign(score=lambda d: d.sv - d.sa)
    mot = S[(S.resp >= 1.0) & (S.resp - S.stim >= 1.0) & S.anat.str.contains(ANAT_MOT)].assign(score=lambda d: d.resp - d.stim)
    neg = S[(S["mean"] <= -0.8) & (S.mx <= 0.5)].assign(score=lambda d: -d["mean"])

    def pick(d, cap):
        out, cnt = [], {}
        for k, r in d.sort_values("score", ascending=False).iterrows():
            if cap is None or cnt.get(r.pat, 0) < cap:
                out.append(k)
                cnt[r.pat] = cnt.get(r.pat, 0) + 1
            if len(out) == per:
                break
        return out
    sel = {"auditory": pick(aud, 3), "visual": pick(vis, None), "motor": pick(mot, 3), "negative": pick(neg, 3)}
    pools = {"auditory": int(len(aud)), "visual": int(len(vis)), "motor": int(len(mot)), "negative": int(len(neg))}
    return sel, pools, S


# ---- the algorithm, exposed --------------------------------------------------------------------
def init_like_pipeline(Xa, k, random_state=0):
    """convex_nmf's initialisation, copied line for line, so one update can be shown in numbers."""
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(random_state)
    n = Xa.shape[0]
    lab = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(Xa)
    G = np.zeros((n, k)) + 0.2
    G[np.arange(n), lab] = 1.0
    W = G / G.sum(0, keepdims=True)
    G = G + 0.2 * rng.random((n, k))
    return W, G, lab


def one_update(Ap, An, W, G):
    """The four matrix products of the W update (Ding, Li & Jordan 2010, eq. 13), and the result."""
    GtG = G.T @ G
    num1, num2 = Ap @ G, An @ W @ GtG
    den1, den2 = An @ G, Ap @ W @ GtG
    W1 = W * np.sqrt((num1 + num2) / np.maximum(den1 + den2, 1e-12))
    return num1, num2, den1, den2, W1


def fit_scale(X, G, comp):
    """Least-squares factor per component so that G @ comp is the converged fit (~1 since the
    2026-09-27 pairing fix in convex_nmf; kept as a guard, computed rather than assumed)."""
    M = (G.T @ G) * (comp @ comp.T)
    b = np.diag(G.T @ X @ comp.T)
    return np.linalg.solve(M, b)


def match_components(comp, means):
    """Permutation of components that best matches the group means, by Pearson r."""
    k = comp.shape[0]
    C = np.array([[np.corrcoef(comp[j], means[g])[0, 1] for g in range(k)] for j in range(k)])
    best = max(permutations(range(k)), key=lambda p: sum(C[j, p[j]] for j in range(k)))
    return list(best), C                     # component j -> group best[j]


def terms_line(pairs, n_show, unit=""):
    """'a×b (name) + a×b (name) + … (m more, summing s)'."""
    shown = pairs[:n_show]
    rest = pairs[n_show:]
    parts = [f"{a:+.2f}×{b:.2f}" + (f" ({nm})" if nm else "") for a, b, nm in shown]
    s = " + ".join(parts).replace("+ -", "− ").replace("+ +", "+ ")
    if s.startswith("+"):
        s = s[1:]
    if rest:
        s += f" + … ({len(rest)} more, summing {sum(a * b for a, b, _ in rest):+.2f})"
    return s


# ---- main -------------------------------------------------------------------------------------
def main(k, n_bins, per):
    Xall, conds, cache_name, n_hg, n_freq = load_cohort(n_bins)
    anat = anatomy()
    S = windows(Xall, n_bins)
    sel, pools, S = select(S, anat, per)
    names = [key for g in GROUPS for key in sel[g]]
    group = [g for g in GROUPS for _ in sel[g]]
    X = Xall.loc[names].to_numpy()
    n, p = X.shape
    gi = {g: [i for i, gg in enumerate(group) if gg == g] for g in GROUPS}
    means = np.array([X[gi[g]].mean(0) for g in GROUPS])
    short = [nm.replace("|", " ") for nm in names]
    anat_s = [short_anat(anat.get(nm, "")) for nm in names]
    h = n_bins // 2

    # the pipeline's fit, with its convergence trace captured from verbose
    buf = io.StringIO()
    with redirect_stdout(buf):
        W, G, comp = LD.convex_nmf(X, k, random_state=0, n_iter=300, verbose=True)
    trace = [(int(m.group(1)), float(m.group(2))) for m in
             re.finditer(r"iter\s+(\d+)\s+\|\|X - GW'X\|\|_F = ([0-9.]+)", buf.getvalue())]
    scale = fit_scale(X, G, comp)
    G = G * scale
    R = G @ comp
    rel = float(np.linalg.norm(X - R) / np.linalg.norm(X))
    A = X @ X.T
    Ap, An = (np.abs(A) + A) / 2.0, (np.abs(A) - A) / 2.0

    # components <-> groups, and the argmax against the groups
    c2g, C = match_components(comp, means)
    tot = G.sum(1)
    share = G / np.maximum(tot[:, None], 1e-12)
    pred = [GROUPS[c2g[int(np.argmax(G[i]))]] for i in range(n)]
    conf = np.array([[sum(1 for i in gi[g] if pred[i] == q) for q in GROUPS] for g in GROUPS])
    acc = int(np.trace(conf))
    comp_of = {GROUPS[c2g[j]]: j for j in range(k)}          # group -> component index

    # ---- the arithmetic, entry by entry (the same lines go on the figure and the site) ----
    ia, la = gi["auditory"][0], gi["auditory"][1]
    others = [i for i in range(n) if group[i] != "auditory"]
    im = min(others, key=lambda i: A[ia, i])                  # the most negative entry of row ia
    ja = comp_of["auditory"]
    t = int(np.argmax(means[0][:h]))                          # the auditory group's peak audio bin
    W0, G0, lab0 = init_like_pipeline(X, k)
    n1, n2, d1, d2, W1 = one_update(Ap, An, W0, G0)
    j0 = int(lab0[ia])
    gram_pos = [(X[ia, q], X[la, q], "") for q in range(p)]
    gram_neg = [(X[ia, q], X[im, q], "") for q in range(p)]
    ct = sorted([(W[i, ja], X[i, t], short[i]) for i in range(n)], key=lambda z: -abs(z[0] * z[1]))
    lines = [
        f"shapes   X {n}×{p} (electrodes × bins)   A = XX′ {n}×{n}   W {n}×{k}   "
        f"F = W′X {k}×{p}   G {n}×{k}   X̂ = G F {n}×{p}",
        "",
        f"Gram, two entries.   A[i,l] = Σ_t X[i,t]·X[l,t]   (sum over the {p} bins)",
        f"  i = {short[ia]}, l = {short[la]} (both auditory):   {terms_line(gram_pos, 3)}",
        f"     = {A[ia, la]:+.1f}   →  A⁺[i,l] = {Ap[ia, la]:.1f},  A⁻[i,l] = {An[ia, la]:.1f}",
        f"  i = {short[ia]}, m = {short[im]} (auditory vs {group[im]}):   {terms_line(gram_neg, 3)}",
        f"     = {A[ia, im]:+.1f}   →  A⁺[i,m] = {Ap[ia, im]:.1f},  A⁻[i,m] = {An[ia, im]:.1f}"
        "      (the sign goes into WHICH half, the size stays)",
        "",
        f"Component, one entry.   F[c_aud, audio bin {t}] = Σ_i W[i, c_aud]·X[i, bin {t}]   "
        f"(W′s column sums to 1, so this is a weighted average of the {n} electrodes)",
        f"  = {terms_line(ct, 4)}  =  {comp[ja, t]:+.2f} dB",
        "",
        f"Reconstruction, one entry.   X̂[i, bin {t}] = Σ_j G[i,j]·F[j, bin {t}]   for i = {short[ia]}",
        "  = " + " + ".join(f"{G[ia, j]:.2f}×{comp[j, t]:+.2f} (c_{GROUPS[c2g[j]][:3]})" for j in range(k))
        + f"  =  {R[ia, t]:+.2f} dB     measured X[i, bin {t}] = {X[ia, t]:+.2f},  residual {X[ia, t] - R[ia, t]:+.2f}",
        "",
        f"One update of one W entry, from the pipeline's initialisation (k-means labels → G = 0.2 + one-hot + noise, W = G / colsum):",
        f"  W[i,j] ← W[i,j] · √( (A⁺G + A⁻W G′G)[i,j] / (A⁻G + A⁺W G′G)[i,j] )   "
        f"for i = {short[ia]}, j = {j0}:",
        f"  = {W0[ia, j0]:.4f} · √( ({n1[ia, j0]:.1f} + {n2[ia, j0]:.1f}) / ({d1[ia, j0]:.1f} + {d2[ia, j0]:.1f}) )"
        f"  =  {W0[ia, j0]:.4f} · {np.sqrt((n1[ia, j0] + n2[ia, j0]) / (d1[ia, j0] + d2[ia, j0])):.3f}  =  {W1[ia, j0]:.4f}"
        "      (A⁺G pulls the weight up where i agrees with what already loads on j; A⁻G pushes it down)",
        "",
        f"Fit after {trace[-1][0] + 1 if trace else '?'} iterations:   ‖X − G W′X‖ / ‖X‖ = {rel:.3f}   "
        f"→  {100 * (1 - rel ** 2):.0f}% of the variance;   argmax(G) puts {acc} of {n} electrodes in their own group.",
    ]

    # ---- figure ---------------------------------------------------------------------------------
    fig = plt.figure(figsize=(16.0, 19.5), dpi=200)
    gs = GridSpec(4, 6, figure=fig, height_ratios=[1.95, 1.30, 1.05, 1.15], hspace=0.50, wspace=0.45)
    fig.suptitle(f"Convex NMF at k = {k} on {n} electrodes chosen to be {WORD.get(len(GROUPS), len(GROUPS))} things  —  "
                 "does  X ≈ G (W′X)  find them, and what is computed on the way?",
                 fontsize=14.5, color=INK, y=0.985)
    fig.text(0.5, 0.965,
             f"{cache_name} · high-gamma ({HG[0]:g}–{HG[1]:g} Hz, {n_hg} of {n_freq} frequency bins) in "
             f"{n_bins} time bins per condition (first {h} = stimulus, last {h} = response) · dB re baseline · "
             f"{per} per group by rule (see the JSON): "
             + ", ".join(f"{g} from {pools[g]} candidates" for g in GROUPS),
             ha="center", fontsize=9.2, color=MUTED)

    # A · X
    ax = fig.add_subplot(gs[0, :4])
    v = np.abs(X).max()
    ax.imshow(X, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto", interpolation="none")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{s}  · {a}" for s, a in zip(short, anat_s)], fontsize=5.2, **MONO)
    for lab_, g in zip(ax.get_yticklabels(), group):
        lab_.set_color(GCOL[g])
    for b in range(1, len(conds)):
        ax.axvline(b * n_bins - 0.5, color=INK, lw=1.4)
    for b in range(len(conds)):
        ax.axvline(b * n_bins + h - 0.5, color=INK, lw=0.6, ls=":")
        ax.text(b * n_bins + n_bins / 2 - 0.5, -1.0, conds[b], ha="center", fontsize=9, color=INK)
        ax.text(b * n_bins + h / 2 - 0.5, n - 0.3, "stim", ha="center", va="top", fontsize=6.5, color=MUTED)
        ax.text(b * n_bins + h + h / 2 - 0.5, n - 0.3, "response", ha="center", va="top", fontsize=6.5, color=MUTED)
    for g in GROUPS[1:]:
        ax.axhline(gi[g][0] - 0.5, color=INK, lw=1.2)
    ax.set_xticks([])
    ax.set_title(f"A · X, {n} × {p}: {per} electrodes of each kind (rows: "
                 + ", ".join(GROUPS) + f"), {n_bins} bins per condition (columns)",
                 fontsize=10.2, color=INK, loc="left", pad=16)

    # B · group means vs components
    ax = fig.add_subplot(gs[0, 4:])
    tt = np.arange(p)
    for gidx, g in enumerate(GROUPS):
        ax.plot(tt, means[gidx], color=GCOL[g], lw=1.4, ls="--", alpha=0.75)
        ax.plot(tt, comp[comp_of[g]], color=GCOL[g], lw=2.3, label=f"component matched to {g}  (r = {C[comp_of[g], gidx]:.2f})")
    ax.axhline(0, color=MUTED, lw=0.7)
    for b in range(1, len(conds)):
        ax.axvline(b * n_bins - 0.5, color=INK, lw=1.0)
    for b in range(len(conds)):
        ax.axvline(b * n_bins + h - 0.5, color=INK, lw=0.5, ls=":")
        ax.text(b * n_bins + n_bins / 2 - 0.5, ax.get_ylim()[1], conds[b], ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_xlim(-0.5, p - 0.5)
    ax.set_xticks([])
    ax.set_ylabel("dB re baseline", fontsize=8.5)
    ax.legend(fontsize=7.2, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.03))
    ax.grid(alpha=0.25)
    ax.set_title(f"B · the {WORD.get(k, k)} components (solid) against each group's mean profile (dashed)",
                 fontsize=10.2, color=INK, loc="left", pad=14)

    # C · W, G, the argmax table
    order = [comp_of[g] for g in GROUPS]                     # columns in group order
    clabels = [f"c{j}→{GROUPS[c2g[j]][:3]}" for j in order]
    for col, (M, cmap, title) in enumerate([(W[:, order], "Purples", "C · W  (columns sum to 1)"),
                                            (G[:, order], "Greens", "G  (graded membership)")]):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=M.max(), aspect="auto", interpolation="none")
        for (r, c), val in np.ndenumerate(M):
            ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=4.8,
                    color=("white" if val > 0.62 * M.max() else INK), **MONO)
        for g in GROUPS[1:]:
            ax.axhline(gi[g][0] - 0.5, color=INK, lw=1.0)
        ax.set_xticks(range(k))
        ax.set_xticklabels(clabels, fontsize=6.0)
        if col == 0:
            ax.set_yticks(range(n))
            ax.set_yticklabels(short, fontsize=4.6, **MONO)
            for lab_, g in zip(ax.get_yticklabels(), group):
                lab_.set_color(GCOL[g])
        else:
            ax.set_yticks([])
        ax.set_title(title, fontsize=9.4, color=INK, loc="left")

    ax = fig.add_subplot(gs[1, 2:4])
    ax.axis("off")
    ax.text(0, 1.00, "argmax(G) against the group each electrode was chosen for", fontsize=9.6, color=INK, va="top")
    ax.text(0.02, 0.86, "chosen as ↓  assigned to →", fontsize=7.6, color=MUTED, va="top")
    cx = [0.52 + 0.15 * q for q in range(k)]
    for q, g in enumerate(GROUPS):
        ax.text(cx[q], 0.86, g, fontsize=7.6, color=GCOL[g], va="top", ha="center")
    for r_, g in enumerate(GROUPS):
        ax.text(0.02, 0.76 - 0.09 * r_, g, fontsize=8, color=GCOL[g], va="top")
        for q in range(k):
            ax.text(cx[q], 0.76 - 0.09 * r_, str(conf[r_, q]), fontsize=9, va="top", ha="center",
                    color=(INK if r_ == q else "#b8860b"), **MONO)
    mixed = [i for i in range(n) if share[i].max() < 0.6]
    ax.text(0, 0.76 - 0.09 * k - 0.04, textwrap.fill(
        f"{acc} of {n} land in their own group. {len(mixed)} electrode{'s' if len(mixed) != 1 else ''} "
        f"ha{'ve' if len(mixed) != 1 else 's'} no majority component"
        + (": " + ", ".join(short[i] for i in mixed) if mixed else "")
        + ". The argmax is only imposed to make this table; G itself keeps the mixture.", 52),
        fontsize=8.2, color=MUTED, va="top", linespacing=1.5)

    ax = fig.add_subplot(gs[1, 4:])
    ax.axis("off")
    ax.text(0, 1.00, "how the components relate to the groups", fontsize=9.6, color=INK, va="top")
    ax.text(0, 0.86, "Pearson r, component (row) × group mean (column)", fontsize=7.6, color=MUTED, va="top")
    cx = [0.26 + 0.16 * q for q in range(k)]
    for q, g in enumerate(GROUPS):
        ax.text(cx[q], 0.74, g, fontsize=7.6, color=GCOL[g], va="top", ha="center")
    for r_, j in enumerate(order):
        ax.text(0.02, 0.64 - 0.09 * r_, f"c{j}", fontsize=8, color=GCOL[GROUPS[c2g[j]]], va="top", **MONO)
        for q in range(k):
            ax.text(cx[q], 0.64 - 0.09 * r_, f"{C[j, q]:+.2f}", fontsize=8.5, va="top", ha="center",
                    color=(INK if c2g[j] == q else MUTED), **MONO)
    ax.text(0, 0.64 - 0.09 * k - 0.04, textwrap.fill(
        "Each component is W′X: a weighted average of real electrodes, in dB. The weights that "
        "build the auditory component sit on the auditory rows of W, and so on — panel C, left.", 60),
        fontsize=8.2, color=MUTED, va="top", linespacing=1.5)

    # D · reconstruction, residual, convergence
    ax = fig.add_subplot(gs[2, :2])
    ax.imshow(R, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto", interpolation="none")
    for b in range(1, len(conds)):
        ax.axvline(b * n_bins - 0.5, color=INK, lw=1.0)
    for g in GROUPS[1:]:
        ax.axhline(gi[g][0] - 0.5, color=INK, lw=1.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("D · X̂ = G (W′X), the reconstruction, same colour scale as A",
                 fontsize=9.8, color=INK, loc="left")
    ax = fig.add_subplot(gs[2, 2:4])
    ax.imshow(X - R, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto", interpolation="none")
    for b in range(1, len(conds)):
        ax.axvline(b * n_bins - 0.5, color=INK, lw=1.0)
    for g in GROUPS[1:]:
        ax.axhline(gi[g][0] - 0.5, color=INK, lw=1.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"X − X̂, what {k} components leave behind\n"
                 f"‖X − X̂‖ / ‖X‖ = {rel:.3f}: {100 * (1 - rel ** 2):.0f}% of the variance kept",
                 fontsize=9.8, color=INK, loc="left")
    ax = fig.add_subplot(gs[2, 4:])
    if trace:
        its, errs = zip(*trace)
        ax.plot(its, errs, color=GREEN, lw=1.8, marker="o", ms=2.5)
        ax.set_xlabel("iteration", fontsize=8)
        ax.set_ylabel("‖X − G W′X‖_F", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    ax.set_title("the multiplicative updates converging\n(stop: relative change < 1e-6, or 300 iterations)",
                 fontsize=9.8, color=INK, loc="left")

    # E · the arithmetic
    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    ax.text(0, 1.02, "E · the arithmetic, entry by entry — every number below is read off the matrices above",
            fontsize=10.2, color=INK, va="top")
    ax.text(0, 0.90, "\n".join(lines), fontsize=6.9, color=INK, va="top", linespacing=1.45, **MONO)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "E14_cnmf_three_roles.png"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    meta_out = dict(
        cache=cache_name, k=k, bins_per_condition=n_bins, per_group=per, conditions=conds,
        groups={g: sel[g] for g in GROUPS}, pools=pools,
        anatomy={nm: anat_s[i] for i, nm in enumerate(names)},
        windows={nm: {c: round(float(S.loc[nm, c]), 3) for c in ("sa", "sv", "sp", "sr", "resp", "stim")} for nm in names},
        rule={"auditory": "sa >= 1.5 dB and sv <= 0.5 dB, superior/transverse temporal, rank sa - sv, <= 3 per patient",
              "visual": "sv >= 0.7, sa <= 0.6, sp and sr >= 0.5, resp <= sv, rank sv - sa, no patient cap",
              "motor": "resp >= 1.0 and resp - stim >= 1.0, pre/postcentral or pars opercularis, rank resp - stim, <= 3 per patient",
              "negative": "mean of the whole profile <= -0.8 dB and max <= 0.5 dB, any anatomy, rank by the mean, <= 3 per patient"},
        n_features=p, n_negative=int((X < 0).sum()), n_values=int(X.size),
        rel_error=round(rel, 4), var_explained=round(1 - rel ** 2, 4), n_iter=(trace[-1][0] + 1 if trace else None),
        component_to_group={int(j): GROUPS[c2g[j]] for j in range(k)},
        corr_component_group=np.round(C, 3).tolist(), confusion=conf.tolist(), argmax_correct=acc,
        mixed=[short[i] for i in mixed],
        mixed_shares={short[i]: {GROUPS[c2g[j]]: round(float(share[i, j]), 3) for j in range(k)} for i in mixed},
        scale_guard=np.round(scale, 6).tolist(),
        X=np.round(X, 3).tolist(), W=np.round(W, 4).tolist(), G=np.round(G, 4).tolist(),
        components=np.round(comp, 3).tolist(), group_means=np.round(means, 3).tolist(),
        trace=trace, arithmetic_lines=lines, electrodes=names, electrode_group=group)
    (OUT / "E14_cnmf_three_roles.json").write_text(json.dumps(meta_out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {png}\n      {OUT / 'E14_cnmf_three_roles.json'}")
    print(f"  {n} electrodes x {p} features, k={k}, rel err {rel:.4f} ({100 * (1 - rel ** 2):.1f}% var), "
          f"argmax {acc}/{n}, scale guard {np.round(scale, 4)}")
    for g in GROUPS:
        print(f"  {g:<9} {pools[g]:3d} candidates -> " + ", ".join(k_.replace('|', ' ') for k_ in sel[g]))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bins", type=int, default=10, help="time bins per condition")
    ap.add_argument("--per", type=int, default=10, help="electrodes per group")
    a = ap.parse_args()
    main(a.k, a.bins, a.per)
