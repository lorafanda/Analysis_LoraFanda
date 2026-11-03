#!/usr/bin/env python3
"""make_member_measures.py - per-electrode membership measures on every feature set (2026-11-03).

For each convex NMF run the walkthrough covers (the paper's four feature sets at their K, the
band-equalised variants at K 7), per electrode: its loading on its own type and the margin to
the next type, its silhouette in the fit's space (unit-norm rows, Euclidean: the paper's
definition), its correlation with its type's mean, its consensus with its own type and with the
best other type from the patient-resampling sweep (mean M over the type's members, Lora's
definition), and its leave-one-patient-out and patient-halves hit rates from the MMD run. A
measure whose source has not been run for a set is left empty and the figure says so.

    python make_member_measures.py                # outputs/clustering/paper2_validation/member_measures/
    python make_member_measures.py --no-brains    # without the glass-brain row (no VTK needed)

Writes member_measures_<set>.csv (one row per electrode), summary.json (per type and per set,
the cross-set rank correlations, the checks) and one figure per set (every electrode a dot,
types in the page's order and colours). The checks compare the main cell with what the story,
the MMD run and the held-out bundle already publish. Reads outputs only.
"""
from __future__ import annotations

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
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_samples

ROOT = Path(__file__).resolve().parent
_sp = importlib.util.spec_from_file_location("p2", ROOT / "00_Paper2_Figures.py")
P2 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(P2)
_dr = importlib.util.spec_from_file_location("p2driver", ROOT / "run_paper_figures_v17.py")
DRV = importlib.util.module_from_spec(_dr)
_dr.loader.exec_module(DRV)
sys.path.insert(0, str(ROOT / "functions"))
import lf_runs as LR                                      # noqa: E402

VARIANTS = ["concat_bands15z", "concat_bands15s", "concat_bands15t"]
FSETS = list(P2.FSETS) + VARIANTS
K_BY_FSET = {**dict(DRV.K), **{f: 7 for f in VARIANTS}}
FS_SHORT = {**P2.FS_SHORT, "concat_bands15z": "15 bands z", "concat_bands15s": "15 bands shape", "concat_bands15t": "15 bands ternary"}
VAL = ROOT / "outputs" / "clustering" / "paper2_validation"
OUT = VAL / "member_measures"
COV = ROOT / "outputs" / "250_recon" / "fsaverage" / "coverage_viz"
SITE_ORDER = Path.home() / "lorafanda.github.io" / "cluster_visualizer_order.json"
INK, MUTED = P2.INK, P2.MUTED
NL = chr(10)
MEASURES = [("loading", "loading on own type"), ("silhouette", "silhouette, unit-norm space"),
            ("r_mean", "r with the type's mean"), ("cons_own", "consensus with own type"),
            ("lopo_hit", "placed in its type, one patient out"), ("halves_hit", "placed in its type, patient halves")]


def hsl2rgb(h, s, l):
    k = lambda n: (n + h * 12) % 12
    a = s * min(l, 1 - l)
    f = lambda n: l - a * max(-1, min(k(n) - 3, min(9 - k(n), 1)))
    return np.array([f(0), f(8), f(4)], float)


def u2hue(pos, K):
    return hsl2rgb(pos / K, 0.62, 0.52)


def unit(X):
    X = np.asarray(X, float)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def keys_of(meta):
    return [f"{p}|{P2.norm(e)}" for p, e in zip(meta["patient_id"], meta["electrode"])]


def figure_order(fset, run, K):
    try:
        o = json.loads(SITE_ORDER.read_text(encoding="utf-8"))["runs"]
        return [int(x) for x in o[f"cnmf__{fset}__{run.name}"][str(K)]]
    except Exception:
        return list(range(K))


def consensus_per_electrode(fset, K, keys, lab):
    """Own-type and best-other consensus per electrode from the sweep's M, with the PUBLISHED labels
    (mean M over the other members of the type): Lora's definition, electrode by electrode."""
    cdir = ROOT / "outputs" / "clustering" / "consensus" / LR.current_cache() / fset / "cnmf"
    it_f, M_f = cdir / f"items_k{K:02d}.csv", cdir / f"M_k{K:02d}.npy"
    if not (it_f.exists() and M_f.exists()):
        return None, None
    it = pd.read_csv(it_f)
    it["key"] = [f"{p}|{P2.norm(e)}" for p, e in zip(it.patient_id, it.electrode)]
    pos = {k: i for i, k in enumerate(it.key)}
    idx = np.array([pos.get(k, -1) for k in keys])
    if (idx < 0).any():
        return None, None
    M = np.load(M_f).astype(float)[np.ix_(idx, idx)]
    n = len(keys); own = np.full(n, np.nan); other = np.full(n, np.nan)
    for c in np.unique(lab):
        m = lab == c
        for i in np.where(m)[0]:
            same = m.copy(); same[i] = False
            own[i] = M[i, same].mean() if same.any() else np.nan
            other[i] = max((M[i, lab == c2].mean() for c2 in np.unique(lab) if c2 != c), default=np.nan)
    return own, other


def heldout_per_electrode(fset, keys):
    cell = ROOT / "outputs" / "clustering" / "mmd" / LR.current_cache() / fset / "cnmf"
    out = {}
    for name in ("lopo", "halves"):
        f = cell / f"confusion_{name}_electrodes.csv"
        if not f.exists():
            out[name] = None; continue
        t = pd.read_csv(f)
        t["key"] = [f"{p}|{P2.norm(e)}" for p, e in zip(t.patient, t.electrode)]
        out[name] = t.set_index("key").hit_rate.reindex(keys).to_numpy(float)
    return out["lopo"], out["halves"]


REF, REF_K = "concat_bands15", 7
_REF = {}


def ref_run():
    """The published run: its labels, its figure order and its type names, loaded once."""
    if not _REF:
        d = P2.load_run(REF, REF_K, "cnmf"); d["fset"] = REF
        _REF.update(lab=np.asarray(d["lab"]), keys=keys_of(d["meta"]), order=figure_order(REF, d["run"], REF_K),
                    names={int(k_): v for k_, v in P2.TYPE_NAMES.get((REF, REF_K), {}).items()})
    return _REF


def match_to_ref(lab, K, keys):
    """Hungarian assignment of this run's clusters to the published types on shared electrodes:
    {cluster: (type, jaccard)}; a cluster without a partner maps to (None, best jaccard)."""
    R = ref_run()
    assert keys == R["keys"], "electrodes differ from the published run"
    M = np.zeros((K, REF_K), int)
    for a, b in zip(lab, R["lab"]):
        M[a, b] += 1
    ri, ci = linear_sum_assignment(-M)
    sa = np.bincount(lab, minlength=K); sb = np.bincount(R["lab"], minlength=REF_K)
    J = M / np.maximum(sa[:, None] + sb[None, :] - M, 1)
    mp = dict(zip(ri.tolist(), ci.tolist()))
    return {a: ((mp[a], float(J[a, mp[a]])) if a in mp else (None, float(J[a].max()))) for a in range(K)}


def measures_for_set(fset):
    """One row per electrode for this set's convex NMF run at the paper's K, or None without a run."""
    K = K_BY_FSET[fset]
    try:
        d = P2.load_run(fset, K, "cnmf")
    except SystemExit as e:
        print(f"  {fset}: {e}")
        return None
    d["fset"] = fset
    X, Gn, lab, meta = d["X"], d["Gn"], d["lab"], d["meta"]
    keys = keys_of(meta)
    n = len(X)
    own = Gn[np.arange(n), lab]
    srt = np.sort(Gn, 1)
    margin = srt[:, -1] - (srt[:, -2] if Gn.shape[1] > 1 else 0)
    sil = silhouette_samples(unit(X), lab)
    C = P2.cube(X, d).reshape(n, -1)
    r_mean = np.full(n, np.nan)
    for c in range(K):
        m = lab == c
        if m.sum() < 2:
            continue
        mu = C[m].mean(0)
        for i in np.where(m)[0]:
            r_mean[i] = np.corrcoef(C[i], mu)[0, 1]
    cons_own, cons_other = consensus_per_electrode(fset, K, keys, lab)
    lopo, halves = heldout_per_electrode(fset, keys)
    if fset == REF:
        order, match = ref_run()["order"], {a: (a, 1.0) for a in range(K)}
    else:
        match = match_to_ref(lab, K, keys)
        rp = {b: p for p, b in enumerate(ref_run()["order"])}
        order = sorted(range(K), key=lambda a: (rp[match[a][0]] if match[a][0] is not None else 99, a))
    df = pd.DataFrame(dict(key=keys, patient=meta.patient_id.to_numpy(), electrode=meta.electrode.to_numpy(),
                           cluster=lab, position=[order.index(int(c)) + 1 if int(c) in order else None for c in lab],
                           matched_type=[match[int(c)][0] for c in lab],
                           loading=own, margin=margin, silhouette=sil, r_mean=r_mean,
                           cons_own=cons_own if cons_own is not None else np.nan,
                           cons_other=cons_other if cons_other is not None else np.nan,
                           lopo_hit=lopo if lopo is not None else np.nan, halves_hit=halves if halves is not None else np.nan))
    names = P2.TYPE_NAMES.get((fset, K), {})
    return dict(fset=fset, k=K, run=d["run"].name, order=order, names={int(k_): v for k_, v in names.items()}, df=df,
                match={int(a): [None if b is None else int(b), float(j)] for a, (b, j) in match.items()},
                xyz=d["xyz"], hemi=d["hemi"])


def summarise(S):
    """Per type and overall: medians, shares, the always / never counts of the held-out placements."""
    df, K = S["df"], S["k"]
    def block(sub):
        o = dict(n=int(len(sub)), loading_median=float(sub.loading.median()), loading_below_cut=float((sub.loading < P2.LOAD_CUT).mean()),
                 loading_at_least_half=float((sub.loading >= 0.5).mean()), silhouette_median=float(sub.silhouette.median()),
                 silhouette_negative=float((sub.silhouette < 0).mean()), r_mean_median=float(sub.r_mean.median()))
        if sub.cons_own.notna().any():
            o.update(cons_own_median=float(sub.cons_own.median()), cons_own_mean=float(sub.cons_own.mean()),
                     cons_margin_median=float((sub.cons_own - sub.cons_other).median()), cons_switch=float((sub.cons_own < sub.cons_other).mean()))
        if sub.lopo_hit.notna().any():
            o.update(lopo_mean=float(sub.lopo_hit.mean()), lopo_always=float((sub.lopo_hit >= 1).mean()), lopo_never=float((sub.lopo_hit <= 0).mean()))
        if sub.halves_hit.notna().any():
            o.update(halves_mean=float(sub.halves_hit.mean()), halves_always=float((sub.halves_hit >= 0.999).mean()), halves_never=float((sub.halves_hit <= 0).mean()))
        return o
    per_type = {int(c): block(df[df.cluster == c]) for c in S["order"]}
    return dict(fset=S["fset"], k=K, run=S["run"], order=S["order"], names=S["names"], match=S["match"], overall=block(df), per_type=per_type,
                has=dict(consensus=bool(df.cons_own.notna().any()), heldout=bool(df.lopo_hit.notna().any())))


# ---------------------------------------------------------------- the brains
_SCENE = {}


def render_view(side, xyz, hemi, cols, rads, alphas, zoom=1.30):
    """One lateral glass view (the types figure's scene) with the contacts whose radius is > 0:
    colours 0-1 RGB, radii in mm, alphas 0-255. Copied from make_walkthrough so the brains here
    are the page's brains."""
    if "d" not in _SCENE:
        _SCENE["d"] = dict(xyz=np.asarray(xyz, float), hemi=np.asarray(hemi))
    pl, ok = P2._scene(side, _SCENE["d"], zoom)
    m = ok & (rads > 0) & np.isfinite(_SCENE["d"]["xyz"][:, 0])
    actor = None
    if m.sum():
        rgba = np.zeros((int(m.sum()), 4), np.uint8)
        rgba[:, :3] = np.clip(255 * np.asarray(cols)[m], 0, 255).astype(np.uint8)
        rgba[:, 3] = np.clip(np.asarray(alphas)[m], 0, 255).astype(np.uint8)
        cloud = P2.pv.PolyData(_SCENE["d"]["xyz"][m])
        cloud["rgba"] = rgba; cloud["r"] = np.asarray(rads)[m].astype(float)
        g = cloud.glyph(orient=False, scale="r", geom=P2.pv.Sphere(radius=1.0, theta_resolution=12, phi_resolution=12))
        actor = pl.add_mesh(g, scalars="rgba", rgba=True)
    img = pl.screenshot(return_img=True, transparent_background=True)
    if actor is not None:
        pl.remove_actor(actor)
    return P2._crop_alpha(np.asarray(img))


def loading_radius(lv):
    """The page's size rule: 1.05 mm at a loading of 0.14, 3.2 mm at 0.85."""
    w = np.clip((np.asarray(lv, float) - 0.14) / (0.85 - 0.14), 0, 1)
    return 1.05 + 2.18 * w


def brain_rows(S, hues, d, weak):
    """Two pairs of views: every electrode in its (matched) type's colour, sized by its loading;
    then the weak members alone, black, over the faint rest."""
    n = len(S["df"]); df = S["df"]
    cols = np.array([hues[int(c)] for c in df.cluster]); rads = loading_radius(df.loading.to_numpy()); al = np.full(n, 230.0)
    pair1 = [render_view(s_, d["xyz"], d["hemi"], cols, rads, al) for s_ in ("L", "R")]
    cols2 = np.where(weak[:, None], np.array([[0.12, 0.12, 0.12]]), np.array([[0.80, 0.81, 0.84]]))
    rads2 = np.where(weak, 2.1, 1.0); al2 = np.where(weak, 255.0, 80.0)
    pair2 = [render_view(s_, d["xyz"], d["hemi"], cols2, rads2, al2) for s_ in ("L", "R")]
    return pair1, pair2


def fig_set(S, out, d=None, brains=True):
    """Top row, one panel per measure: every electrode a dot, a violin per type, the median as a
    bar; hit rates as stacked bars (always / sometimes / never placed in its type). Bottom row,
    when brains are drawn: the electrodes on the glass brains in their type's colour sized by
    loading, and the weak members (loading below the draw cut, or a negative silhouette) alone."""
    import textwrap
    df, K, order, names = S["df"], S["k"], S["order"], S["names"]
    R = ref_run(); rp = {b: p for p, b in enumerate(R["order"])}
    main = S["fset"] == REF
    hues = {j: (u2hue(p, K) if main else (u2hue(rp[S["match"][j][0]], REF_K) if S["match"][j][0] is not None else np.array([0.6, 0.6, 0.6])))
            for p, j in enumerate(order)}
    tick = (lambda j: f"c{j}") if main else (lambda j: f"v{j}" + NL + (f"→c{S['match'][j][0]}" if S["match"][j][0] is not None else "→ –"))
    rng = np.random.default_rng(0)
    weak = (df.loading.to_numpy() < P2.LOAD_CUT) | (df.silhouette.to_numpy() < 0)
    nrow = 2 if (brains and d is not None) else 1
    fig = plt.figure(figsize=(3.3 * len(MEASURES), 4.6 * nrow + 0.6), dpi=150)
    gs = fig.add_gridspec(nrow, len(MEASURES), height_ratios=[1, 0.95][:nrow], hspace=0.55, wspace=0.38, top=0.86 if nrow == 2 else 0.80, bottom=0.08)
    for i_, (col, title) in enumerate(MEASURES):
        ax = fig.add_subplot(gs[0, i_])
        ax.set_title(title, fontsize=9, loc="left")
        if df[col].isna().all():
            ax.text(0.5, 0.5, "not run yet" + NL + "for this set", ha="center", va="center", fontsize=9, color=MUTED, transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            continue
        if col in ("lopo_hit", "halves_hit"):
            for p, j in enumerate(order):
                v = df.loc[df.cluster == j, col].dropna().to_numpy()
                if not len(v):
                    continue
                always = float((v >= 0.999).mean()); never = float((v <= 0).mean()); mid = 1 - always - never
                ax.bar(p, always, color=hues[j], width=0.72)
                ax.bar(p, mid, bottom=always, color=hues[j], alpha=0.4, width=0.72)
                ax.bar(p, never, bottom=always + mid, color="#d9d9d9", width=0.72)
                ax.plot(p, float(v.mean()), "o", color=INK, ms=5, zorder=3)
            ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "50", "100 %"], fontsize=8)
            ax.set_xlabel("full always · pale sometimes · grey never · dot mean", fontsize=6.8, color=MUTED, labelpad=3)
        else:
            data = [df.loc[df.cluster == j, col].dropna().to_numpy() for j in order]
            parts = ax.violinplot([v if len(v) > 1 else np.r_[v, v] for v in data], positions=range(K), showmedians=False, showextrema=False, widths=0.9)
            for body, j in zip(parts["bodies"], order):
                body.set_facecolor(hues[j]); body.set_alpha(0.25); body.set_edgecolor("none")
            for p, (j, v) in enumerate(zip(order, data)):
                ax.scatter(p + rng.uniform(-0.26, 0.26, len(v)), v, s=11, color=hues[j], alpha=0.6, lw=0.3, edgecolor="white")
                ax.plot([p - 0.36, p + 0.36], [np.median(v)] * 2, color=INK, lw=2.2, zorder=3)
            if col == "loading":
                ax.axhline(P2.LOAD_CUT, ls="--", lw=0.9, color=MUTED); ax.axhline(0.5, ls=":", lw=0.9, color=MUTED); ax.set_ylim(0, 1.02)
            elif col == "cons_own":
                ax.set_ylim(0, 1.02)
            elif col == "silhouette":
                ax.axhline(0, lw=0.9, color=MUTED)
            elif col == "r_mean":
                ax.axhline(0, lw=0.9, color=MUTED); ax.set_ylim(-1.02, 1.02)
            ax.tick_params(axis="y", labelsize=8)
        ax.set_xticks(range(K)); ax.set_xticklabels([tick(j) for j in order], fontsize=7.5)
        for tl, j in zip(ax.get_xticklabels(), order):
            tl.set_color(hues[j])
        ax.tick_params(length=2); ax.spines[["top", "right"]].set_visible(False)
    if nrow == 2:
        pair1, pair2 = brain_rows(S, hues, d, weak)
        sub = gs[1, :].subgridspec(1, 4, wspace=0.06)
        titles = {0: "every electrode in its type's colour, sized by its loading (the page's rule)",
                  2: f"the weak members alone: {int(weak.sum())} electrodes with loading < {P2.LOAD_CUT:.1f} or silhouette < 0"}
        for k_, (img, side) in enumerate([(pair1[0], "L"), (pair1[1], "R"), (pair2[0], "L"), (pair2[1], "R")]):
            ax = fig.add_subplot(sub[0, k_])
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            ax.text(0.02, 0.02, side, transform=ax.transAxes, fontsize=8, color=MUTED, ha="left", va="bottom")
            if k_ in titles:
                ax.set_title(titles[k_], fontsize=8.5, loc="left", color=INK)
    if main:
        legend = "   ".join(f"c{j} {names[j]}" for j in order if j in names)
    else:
        legend = "   ".join((f"v{j} → c{S['match'][j][0]} {R['names'].get(S['match'][j][0], '')} (J {S['match'][j][1]:.2f})" if S["match"][j][0] is not None
                             else f"v{j} → no partner") for j in order)
        legend = "v = this run's own cluster id, ordered and coloured by the published type it is matched to (Hungarian on shared electrodes):   " + legend
    fig.suptitle(f"{FS_SHORT.get(S['fset'], S['fset'])} × convex NMF   ·   K {K}   ·   run {S['run']}   ·   every electrode a dot, the bar its cluster's median"
                 + (NL + textwrap.fill(legend, 210) if legend else ""), fontsize=9.5, x=0.01, ha="left", y=0.995)
    p = out / f"member_measures_{S['fset']}.png"
    fig.savefig(p, facecolor="white", bbox_inches="tight"); plt.close(fig)
    return p


def checks(sets):
    """The main cell against what is already published: the story's loading numbers, the MMD
    run's per-type silhouettes, the held-out bundle's per-type consensus and its always / never counts."""
    out = []
    main = next((S for S in sets if S["fset"] == "concat_bands15"), None)
    if main is None:
        return out
    df = main["df"]
    def chk(name, got, want, tol):
        ok = (got is not None and want is not None and abs(got - want) <= tol)
        out.append(dict(check=name, got=got, want=want, ok=bool(ok)))
        print(f"  {'MATCH   ' if ok else 'MISMATCH'} {name}: got {got}, published {want}")
    nums = json.loads((VAL / "numbers.json").read_text(encoding="utf-8"))["values"]
    chk("median own loading (story)", round(float(df.loading.median()), 3), round(float(nums["own_median"]), 3), 0.0015)
    chk("share below the draw cut (story)", round(float((df.loading < P2.LOAD_CUT).mean()), 3), round(float(nums["own_below_cut"]), 3), 0.0015)
    cell = ROOT / "outputs" / "clustering" / "mmd" / LR.current_cache() / "concat_bands15" / "cnmf"
    if (cell / "per_type.csv").exists():
        pt = pd.read_csv(cell / "per_type.csv"); pt = pt[pt.version == "argmax"].set_index("cluster")
        for c in main["order"]:
            chk(f"silhouette mean c{c} (MMD run)", round(float(df.loc[df.cluster == c, "silhouette"].mean()), 3), round(float(pt.loc[c, "silhouette"]), 3), 0.002)
    ho_f = COV / "runs" / f"cnmf__concat_bands15__{main['run']}" / "heldout.json"
    if ho_f.exists():
        ho = json.loads(ho_f.read_text(encoding="utf-8"))
        T = {t["cluster"]: t for t in ho["types"]}
        for c in main["order"]:
            chk(f"consensus of c{c} (held-out bundle)", round(float(df.loc[df.cluster == c, "cons_own"].mean()), 2), round(float(T[c]["consensus"]), 2), 0.011)
        chk("electrodes always placed right, one patient out (held-out bundle)", int((df.lopo_hit >= 1).sum()), sum(1 for v in ho["contacts"]["lopo_hit"] if v is not None and v >= 1), 0)
        chk("electrodes never placed right, one patient out (held-out bundle)", int((df.lopo_hit <= 0).sum()), sum(1 for v in ho["contacts"]["lopo_hit"] if v is not None and v <= 0), 0)
    return out


def cross_set(sets):
    """Spearman rank correlation of each per-electrode measure between sets (same electrodes, same order)."""
    out = {}
    for col in ("loading", "silhouette", "r_mean", "cons_own", "lopo_hit"):
        names = [S["fset"] for S in sets if S["df"][col].notna().any()]
        M = {}
        for a in names:
            M[a] = {}
            for b in names:
                x = next(S for S in sets if S["fset"] == a)["df"][col].to_numpy(float)
                y = next(S for S in sets if S["fset"] == b)["df"][col].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                M[a][b] = float(spearmanr(x[ok], y[ok]).correlation) if ok.sum() > 10 else None
        out[col] = M
    return out


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try: stream.reconfigure(errors="replace")
            except Exception: pass
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--no-brains", action="store_true", help="skip the glass-brain row (no VTK; a locked screen cannot render)")
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    sets, summaries, figs = [], [], {}
    keys0 = None
    for fset in FSETS:
        S = measures_for_set(fset)
        if S is None:
            continue
        if keys0 is None:
            keys0 = list(S["df"].key)
        elif list(S["df"].key) != keys0:
            raise SystemExit(f"{fset}: electrodes differ from {sets[0]['fset']}; the cross-set tables would lie")
        sets.append(S)
        S["df"].to_csv(OUT / f"member_measures_{fset}.csv", index=False)
        summaries.append(summarise(S))
        figs[fset] = fig_set(S, out=OUT, d=S, brains=not a.no_brains).name
        o = summaries[-1]["overall"]
        print(f"  {fset:16s} K {S['k']}  run {S['run']}  loading median {o['loading_median']:.2f}  < cut {100 * o['loading_below_cut']:.0f} %  "
              f"silhouette median {o['silhouette_median']:+.3f}  negative {100 * o['silhouette_negative']:.0f} %"
              + (f"  consensus median {o['cons_own_median']:.2f}" if 'cons_own_median' in o else "  consensus -")
              + (f"  LOPO always {100 * o['lopo_always']:.0f} % never {100 * o['lopo_never']:.0f} %" if 'lopo_always' in o else "  held-out -"))
    print("checks against the published numbers (main cell):")
    ck = checks(sets)
    xs = cross_set(sets)
    (OUT / "summary.json").write_text(json.dumps(dict(cache=LR.current_cache(), built=time.strftime("%Y-%m-%d %H:%M"), sets=summaries, figures=figs,
                                                      checks=ck, cross_set=xs, not_run=[f for f in FSETS if f not in figs]), indent=1), encoding="utf-8")
    print(f"{len(sets)} sets -> {OUT}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
