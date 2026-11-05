#!/usr/bin/env python3
"""project_ungated.py - the electrodes the gate removed, projected onto the published archetypes without
a refit (Lora, 2026-11-05).

    python project_ungated.py                     # NiMARE maps (member_measures/0_nimare_maps.py), loading 0.3
    python project_ungated.py --ns fdr            # the FDR-thresholded NeuroSynth maps instead
    python project_ungated.py --loading 0.5

THE PROJECTION. The published run (convex NMF, 15 bands, K 7) fixes seven archetypes C (components_by_k/
C_k07.npy, in the unit-norm space the fit used). Every electrode the pipeline saw on cohort v17 - the 1587
that passed the response gate and the 1330 that did not - is built exactly as 240 builds it (the same cache,
the same 15-band features, unit length), and its loadings on the seven archetypes are the non-negative least
squares solution g >= 0 of x ~ g C. Normalised as the pipeline normalises (rows of G to sum 1): the type is
the argmax, the loading its value, and r2 = 1 - |x - gC|^2 is how much of the electrode's shape the seven
archetypes explain. The gated electrodes are projected the same way, so the projection can be checked
against the run's own loadings before the ungated ones are read.

THE MAPS. Every electrode is also sampled on the NeuroSynth maps the way member_measures/4_neurosynth.py
samples the gated cohort (--ns nimare: the mean z in a 5-mm sphere, inside = the map's top 10 %; --ns fdr:
the nearest voxel, inside = z > 0). Per type, the share inside each map is given for the gated members and for
the ungated electrodes the projection assigns to the type, both against the whole seen set as baseline, with
a patient bootstrap. No spatial null.

A credible ungated member needs loading >= --loading AND r2 >= --r2 (0.5): the archetypes must explain half of its
shape. The null for r2: every electrode with each condition's time axis rolled by a random offset, projected the same way.
Writes outputs/clustering/ungated_projection/<cache>/: projection.csv (one row per electrode), maps_inside.csv,
summary.md, fig1_projection.png, fig2_maps.png. Reads only. Local: the cache is on the share, no server."""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "functions"))
import lf_concat as CC                                    # noqa: E402
import lf_runs as LR                                      # noqa: E402
import lf_decompose as D                                  # noqa: E402
os.environ["PAPER2_PAPER"] = "0"
_sp = importlib.util.spec_from_file_location("p2", ROOT / "00_Paper2_Figures.py")
P2 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(P2)

REF, K = "concat_bands15", 7
INK, MUTED, GREY = P2.INK, P2.MUTED, P2.GREY
NL, BS = chr(10), chr(92)
INPUT_DIR = Path(BS * 2 + BS.join(["nasac-m2.unige.ch", "m-HumanNeuronLab", "ANALYSIS", "FLM", "Analysis_LoraFanda",
                                   "01_FBM_Analysis", "outputs", "03_ERSP"]))   # the cache key carries this spelling (240)
POOL = ROOT.parent / "04_FBM_Pooling"
MNI305_TO_152 = np.array([[0.9975, -0.0073, 0.0176, -0.0429], [0.0146, 1.0009, -0.0024, 1.5496],
                          [-0.0130, -0.0093, 0.9971, 1.1840], [0.0, 0.0, 0.0, 1.0]])
THR_DB, MIN_PROP = 2.2, 0.02                              # the gate's positive rule (cache params): descriptive axis only

ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
ap.add_argument("--ns", default="nimare", choices=["nimare", "fdr"])
ap.add_argument("--loading", type=float, default=0.30, help="a projected electrode belongs to a type at or above this normalised loading")
ap.add_argument("--boots", type=int, default=1000)
ap.add_argument("--r2", type=float, default=0.50, help="an ungated electrode is a credible member only if the archetypes also explain this much of its shape")
a = ap.parse_args()
L = a.loading
R2MIN = a.r2
OUT = ROOT / "outputs" / "clustering" / "ungated_projection" / LR.current_cache()
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. the run and its archetypes
run = LR.newest_run("cnmf", REF)
C = np.load(run / "components_by_k" / f"C_k{K:02d}.npy").astype(float)             # K x features, unit-norm space
G = np.load(run / "loadings_by_k" / f"G_k{K:02d}.npy").astype(float)
Gn_run = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
Xtr = np.load(run / "X_train.npy").astype(float)
meta = pd.read_csv(run / "labels.csv")
keys_run = [f"{p}|{CC.normalize_label(e)}" for p, e in zip(meta.patient_id.astype(str), meta.electrode)]
names = {j: P2.type_name(REF, K, j) for j in range(K)}
d_ref = P2.load_run(REF, K, "cnmf")
means = np.stack([P2.cube(d_ref["X"], d_ref)[d_ref["lab"] == j].mean(0) for j in range(K)])
ORDER = [int(j) for j in P2.block_order(means, d_ref)[0]]                                   # the paper's order
COL = {j: P2.cluster_col(j, K) for j in range(K)}
print(f"run {run.name}: {len(Xtr)} gated electrodes, K {K}, archetypes {C.shape}")

# ---------------------------------------------------------------- 2. every electrode the pipeline saw, built as 240 builds it
df, Xc = CC.build_concat_dataset(INPUT_DIR, conditions=("audio", "picture", "reading"), require_high_activity=False,
                                 cache_dir=CC.DEFAULT_CONCAT_CACHE, verbose=False)
Xall = CC.concat_bands15_features(Xc, n_blocks=3, fmax_hz=398.4375).astype(float)
keys = [f"{p}|{c}" for p, c in zip(df.patient_id.astype(str), df.contact_norm)]
pat = df.patient_id.astype(str).to_numpy()
pos = {k: i for i, k in enumerate(keys)}
missing = [k for k in keys_run if k not in pos]
if missing:
    raise SystemExit(f"{len(missing)} gated electrodes of the run are not in the ungated build: {missing[:5]}")
idx = np.array([pos[k] for k in keys_run])
dmax = float(np.abs(Xall[idx] - Xtr).max())
if dmax > 1e-3:
    raise SystemExit(f"the rebuilt features differ from the run's X_train (max |diff| {dmax:.4g}): the chain is not the run's")
gated = np.zeros(len(keys), bool); gated[idx] = True
n_hi = df.n_high_activity.to_numpy(int) if "n_high_activity" in df.columns else np.where(gated, 1, 0)
if not np.array_equal(n_hi > 0, gated):
    print(f"  NOTE: n_high_activity > 0 disagrees with the run's membership for {int((n_hi > 0) != gated).sum()} electrodes")
cm = pd.read_parquet(Path(CC.DEFAULT_CONCAT_CACHE) / 'df_meta.parquet')
cm['key'] = cm.patient_id.astype(str) + '|' + cm['electrode'].map(CC.normalize_label)
gstat = cm.groupby('key').agg(prop=('prop_above_pos', 'max'), thr=('thr_pos_used', 'max'), neg=('prop_below_neg', 'max')).reindex(keys)
frac = gstat.prop.to_numpy(float)                      # the gate's positive statistic: the largest share of bins above the patient's threshold
print(f"seen {len(keys)} electrodes in {len(set(pat))} patients: {int(gated.sum())} gated, {int((~gated).sum())} not; "
      f"rebuilt features match X_train to {dmax:.2g}")

# ---------------------------------------------------------------- 3. the projection, gated and ungated alike
Xs = D.unit_norm(Xall)
A = C.T
Gp = np.array([nnls(A, x)[0] for x in Xs])
r2 = 1.0 - ((Xs - Gp @ C) ** 2).sum(1)
Gpn = Gp / np.maximum(Gp.sum(1, keepdims=True), 1e-12)
type_p, load_p = Gpn.argmax(1), Gpn.max(1)
type_run = np.full(len(keys), -1); type_run[idx] = Gn_run.argmax(1)
load_run = np.full(len(keys), np.nan); load_run[idx] = Gn_run.max(1)
own_run = np.full(len(keys), np.nan); own_run[idx] = Gn_run[np.arange(len(idx)), Gn_run.argmax(1)]
agree = float((type_p[idx] == Gn_run.argmax(1)).mean())
agree_sure = float((type_p[idx] == Gn_run.argmax(1))[Gn_run.max(1) >= L].mean())
r_load = float(np.corrcoef(load_p[idx], Gn_run.max(1))[0, 1])
ve_proj_gated = float(1.0 - ((Xs[idx] - Gp[idx] @ C) ** 2).sum() / (Xs[idx] ** 2).sum())
print(f"projection check on the gated: argmax agrees with the run for {100 * agree:.1f} % (at loading >= {L}: {100 * agree_sure:.1f} %), r(loading) {r_load:.2f}; variance explained {100 * ve_proj_gated:.1f} %")

tab = pd.DataFrame(dict(key=keys, patient=pat, gated=gated, n_high_activity=n_hi, prop_above_pos=np.round(frac, 4), thr_pos_used=gstat.thr.to_numpy(float), prop_below_neg=np.round(gstat.neg.to_numpy(float), 4),
                        type_proj=type_p, loading_proj=np.round(load_p, 4), r2_proj=np.round(r2, 4),
                        type_run=type_run, loading_run=np.round(load_run, 4)))
for j in range(K):
    tab[f"g{j}"] = np.round(Gpn[:, j], 4)
tab.to_csv(OUT / "projection.csv", index=False)
# a null for the fit: the same electrodes with each condition's time axis rolled by a random offset (the shape keeps
# its smoothness and its bands and loses its alignment to the task), projected the same way
nt = Xc.shape[2] // 3
Xsh = np.empty_like(Xc)
for i in range(len(Xc)):
    for b in range(3):
        s_ = int(rng.integers(nt // 5, nt - nt // 5))
        Xsh[i, :, b * nt:(b + 1) * nt] = np.roll(Xc[i, :, b * nt:(b + 1) * nt], s_, axis=1)
Xs_sh = D.unit_norm(CC.concat_bands15_features(Xsh, n_blocks=3, fmax_hz=398.4375).astype(float))
del Xsh
Gsh = np.array([nnls(A, x)[0] for x in Xs_sh])
r2_sh = 1.0 - ((Xs_sh - Gsh @ C) ** 2).sum(1)
load_sh = (Gsh / np.maximum(Gsh.sum(1, keepdims=True), 1e-12)).max(1)
tab["loading_shifted"] = np.round(load_sh, 4); tab["r2_shifted"] = np.round(r2_sh, 4)
tab.to_csv(OUT / "projection.csv", index=False)
ug = ~gated
S = []
S.append(f"# The ungated electrodes projected onto the seven archetypes, cohort v17 (run {run.name}, {a.ns} maps, loading {L})")
S.append("")
S.append(f"Seen {len(keys)} electrodes in {len(set(pat))} patients: {int(gated.sum())} gated, {int(ug.sum())} not. The rebuilt features match the run's X_train to {dmax:.2g}.")
S.append(f"Projection check on the gated electrodes: the argmax agrees with the run's for {100 * agree:.1f} % of them ({100 * agree_sure:.1f} % of those at loading >= {L}); r between the projected and the fitted loading {r_load:.2f}; "
         f"variance explained {100 * ve_proj_gated:.1f} %. The projection is the pipeline's own held-out rule (lf_decompose.cv_rank_curve: non-negative least squares onto the fitted components).")
S.append("")
S.append("| set | n | median loading | share >= 0.3 | share >= 0.5 | median r2 | share r2 >= 0.5 |")
S.append("|---|---|---|---|---|---|---|")
for nm, lo_a, r_a, m in (("gated, the run's own G", own_run, r2, gated), ("gated, projected", load_p, r2, gated), ("ungated, projected", load_p, r2, ug),
                         ("gated, time-shifted null", load_sh, r2_sh, gated), ("ungated, time-shifted null", load_sh, r2_sh, ug)):
    lo = lo_a[m]
    rr = r_a[m]
    S.append(f"| {nm} | {int(m.sum())} | {np.nanmedian(lo):.2f} | {100 * np.nanmean(lo >= 0.3):.0f} % | {100 * np.nanmean(lo >= 0.5):.0f} % | "
             f"{np.median(rr):.2f} | {100 * np.mean(rr >= 0.5):.0f} % |")

# per type: the gated members against the ungated electrodes the projection gives the type
S.append("")
S.append(f"| type | gated members (argmax) | of them at loading >= {L} | ungated at loading >= {L} | their patients | of them with r2 >= {R2MIN} | ungated at loading >= 0.5 | ungated argmax |")
S.append("|---|---|---|---|---|---|---|---|")
per_type = []
for j in ORDER:
    gm = gated & (type_run == j); gm_sure = gm & (own_run >= L)
    um = ug & (type_p == j); um_sure = um & (load_p >= L); um_05 = um & (load_p >= 0.5); um_fit = um_sure & (r2 >= R2MIN)
    per_type.append(dict(type=j, name=names[j], gated=int(gm.sum()), gated_sure=int(gm_sure.sum()), ungated_sure=int(um_sure.sum()),
                         ungated_patients=int(len(set(pat[um_sure]))), ungated_fit=int(um_fit.sum()), ungated_05=int(um_05.sum()), ungated_argmax=int(um.sum())))
    S.append(f"| c{j} {names[j]} | {int(gm.sum())} | {int(gm_sure.sum())} | {int(um_sure.sum())} | {len(set(pat[um_sure]))} | {int(um_fit.sum())} | {int(um_05.sum())} | {int(um.sum())} |")
PT = pd.DataFrame(per_type)
# how the projection reads against the gate's own statistic
bins = [0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 1.0]
S.append("")
S.append("| largest share of bins above threshold | n | gated | median loading | share >= 0.3 | median r2 |")
S.append("|---|---|---|---|---|---|")
for lo_, hi_ in zip(bins[:-1], bins[1:]):
    m = (frac >= lo_) & (frac < hi_)
    if m.sum():
        S.append(f"| {100 * lo_:g} to {100 * hi_:g} % | {int(m.sum())} | {100 * gated[m].mean():.0f} % | {np.median(load_p[m]):.2f} | {100 * np.mean(load_p[m] >= 0.3):.0f} % | {np.median(r2[m]):.2f} |")

# ---------------------------------------------------------------- 4. the maps at every seen electrode (4_neurosynth.py's sampling)
if a.ns == "nimare":
    NS, PATTERN, SPHERE = POOL / "neurosynth_nimare", "*association-test_z.nii.gz", 5.0
    mtab = pd.read_csv(NS / "maps.csv")
    MAPS = [(r.name, r.stage, r.kind) for r in mtab.itertuples()]
    STAGES = ["Picture naming", "Auditory definition", "Sentence completion", "Shared core", "Language network"]
    MAPS.sort(key=lambda m: (STAGES.index(m[1]) if m[1] in STAGES else 9, m[2] != "term"))      # by stage, terms before contrasts
else:
    NS, PATTERN, SPHERE = POOL / "neurosynth", "*association-test_z_FDR*.nii.gz", 0.0
    MAPS = [(p.name, "", "term") for p in sorted(NS.iterdir()) if p.is_dir() and not p.name.startswith("_")]
co = pd.read_csv(ROOT / "outputs" / "250_recon" / "fsaverage" / "coords" / "ALL_PATIENTS_contacts_fsaverage.csv")
co["key"] = [f"{p}|{CC.normalize_label(n)}" for p, n in zip(co.patient, co.name)]
xyz = co.drop_duplicates("key").set_index("key").reindex(keys)[["x", "y", "z"]].to_numpy(float)
okc = np.isfinite(xyz).all(1)
mni = (np.c_[xyz, np.ones(len(xyz))] @ MNI305_TO_152.T)[:, :3]
print(f"maps: {len(MAPS)} ({a.ns}); electrodes with coordinates {int(okc.sum())} of {len(keys)}")


def sample(vol, aff, point):
    inv = np.linalg.inv(aff)
    c = (inv @ np.r_[point, 1.0])[:3]
    if SPHERE <= 0:
        ijk = np.round(c).astype(int)
        if (ijk >= 0).all() and (ijk < np.array(vol.shape[:3])).all():
            return vol[ijk[0], ijk[1], ijk[2]]
        return np.nan
    vs = np.abs(np.diag(aff)[:3])
    r = np.ceil(SPHERE / vs).astype(int)
    lo = np.maximum(np.floor(c).astype(int) - r, 0)
    hi = np.minimum(np.floor(c).astype(int) + r + 1, np.array(vol.shape[:3]))
    if (hi <= lo).any():
        return np.nan
    gi, gj, gk = np.meshgrid(*[np.arange(lo[d_], hi[d_]) for d_ in range(3)], indexing="ij")
    d2 = sum(((g_ - c[d_]) * vs[d_]) ** 2 for d_, g_ in enumerate((gi, gj, gk)))
    sel = d2 <= SPHERE ** 2
    vals = vol[gi[sel], gj[sel], gk[sel]]
    vals = vals[vals != 0] if a.ns == "nimare" else vals
    return float(vals.mean()) if len(vals) else np.nan


INSIDE = {}
for name, stage, kind in MAPS:
    sub = NS / name
    fs = sorted(f for f in sub.glob(PATTERN) if a.ns == "nimare" or "FDR" in f.name)
    if not fs:
        print(f"  {name}: no map file, skipped"); continue
    img = nib.load(str(fs[0])); vol = np.asarray(img.dataobj, float); aff = img.affine
    zs = np.array([sample(vol, aff, mni[i]) if okc[i] else np.nan for i in range(len(keys))])
    if a.ns == "nimare":
        inmask = np.sort(vol[vol != 0])
        pct = 100.0 * np.searchsorted(inmask, zs, side="right") / len(inmask)
        INSIDE[name] = np.where(np.isfinite(zs), pct > 90.0, False)
    else:
        INSIDE[name] = np.where(np.isfinite(zs), zs > 0, False)
MAPS = [m for m in MAPS if m[0] in INSIDE]
print(f"sampled {len(MAPS)} maps")

# ---------------------------------------------------------------- 5. per type and map: gated members vs the projected ungated, one baseline
pats = np.unique(pat); pidx = pd.Series(pat).map({p: i for i, p in enumerate(pats)}).to_numpy()
BOOT = [rng.integers(0, len(pats), len(pats)) for _ in range(a.boots)]
rows = []
for j in ORDER:
    groups = (("gated", gated & (type_run == j) & (own_run >= L)), ("ungated", ug & (type_p == j) & (load_p >= L)),
              ("wellfit", ug & (type_p == j) & (load_p >= L) & (r2 >= R2MIN)))
    for name, stage, kind in MAPS:
        ins = INSIDE[name].astype(float)
        base = ins.mean()                                            # every seen electrode
        Sb = np.bincount(pidx, weights=ins, minlength=len(pats)); Nb = np.bincount(pidx, minlength=len(pats)).astype(float)
        for gname, m in groups:
            n_ = int(m.sum())
            share = ins[m].mean() if n_ else np.nan
            Sm = np.bincount(pidx[m], weights=ins[m], minlength=len(pats)); Nm = np.bincount(pidx[m], minlength=len(pats)).astype(float)
            bs = [(Sm[b].sum() / Nm[b].sum()) / (Sb[b].sum() / Nb[b].sum()) for b in BOOT if Nm[b].sum() and Sb[b].sum()]
            lo, hi = (np.quantile(bs, [0.025, 0.975]) if len(bs) > 50 else (np.nan, np.nan))
            rows.append(dict(type=j, name=names[j], group=gname, n=n_, patients=int(len(set(pat[m]))), map=name, stage=stage, kind=kind,
                             pct_inside=100 * share, baseline_pct=100 * base, enrichment=share / base if base else np.nan,
                             lo=lo, hi=hi, outlined=bool(np.isfinite(lo) and lo > 1)))
MI = pd.DataFrame(rows)
MI.round(3).to_csv(OUT / "maps_inside.csv", index=False)
# how alike the two profiles are, per type, across maps
S.append("")
S.append(f"| type | gated members | ungated projected | r across maps | outlined in both | well-fit ungated (r2 >= {R2MIN}) | r across maps | outlined in both | maps outlined, gated |")
S.append("|---|---|---|---|---|---|---|---|---|")
for j in ORDER:
    g_ = MI[(MI.type == j) & (MI.group == "gated")].set_index("map")
    og = set(g_.index[g_.outlined])
    cells = [f"c{j} {names[j]}", str(int(g_.n.iloc[0]))]
    for gname in ("ungated", "wellfit"):
        u_ = MI[(MI.type == j) & (MI.group == gname)].set_index("map").reindex(g_.index)
        ok_ = g_.pct_inside.notna() & u_.pct_inside.notna()
        r_ = float(np.corrcoef(g_.pct_inside[ok_], u_.pct_inside[ok_])[0, 1]) if ok_.sum() > 3 and g_.pct_inside[ok_].std() > 0 and u_.pct_inside[ok_].std() > 0 else np.nan
        ou = set(u_.index[u_.outlined.fillna(False).astype(bool)])
        cells += [str(int(u_.n.iloc[0])), f"{r_:.2f}", ", ".join(sorted(ou & og)) or "-"]
    S.append("| " + " | ".join(cells) + f" | {', '.join(sorted(og)) or '-'} |")
(OUT / "summary.md").write_text(NL.join(S) + NL, encoding="utf-8")
print(NL.join(S))

# ---------------------------------------------------------------- 6. figure 1: the projection itself
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), dpi=170, gridspec_kw=dict(width_ratios=[1, 1, 1.25, 1.35]))
ax = axes[0]
ax.hist(own_run[gated], bins=np.linspace(1 / K, 1, 36), color=INK, alpha=0.85, label="gated, the run's G", density=True)
ax.hist(load_p[gated], bins=np.linspace(1 / K, 1, 36), histtype="step", color=INK, lw=1.2, label="gated, projected", density=True)
ax.hist(load_p[ug], bins=np.linspace(1 / K, 1, 36), color="#c1121f", alpha=0.45, label="ungated, projected", density=True)
ax.axvline(L, color=MUTED, ls=(0, (4, 3)), lw=1); ax.set_xlabel("loading on the type (largest normalised loading)", fontsize=8)
ax.set_title("A  how much of one type an electrode is", fontsize=9, loc="left"); ax.legend(frameon=False, fontsize=7); ax.set_yticks([])
ax = axes[1]
ax.hist(r2[gated], bins=np.linspace(-0.2, 1, 40), color=INK, alpha=0.85, label="gated", density=True)
ax.hist(r2[ug], bins=np.linspace(-0.2, 1, 40), color="#c1121f", alpha=0.45, label="ungated", density=True)
ax.hist(r2_sh[gated], bins=np.linspace(-0.2, 1, 40), histtype="step", color=INK, lw=1.0, ls=(0, (3, 2)), label="gated, time-shifted", density=True)
ax.hist(r2_sh[ug], bins=np.linspace(-0.2, 1, 40), histtype="step", color="#c1121f", lw=1.0, ls=(0, (3, 2)), label="ungated, time-shifted", density=True)
ax.set_xlabel("r2: the shape explained by the seven archetypes", fontsize=8); ax.set_yticks([]); ax.legend(frameon=False, fontsize=7)
ax.set_title("B  how well the archetypes fit the shape", fontsize=9, loc="left")
ax = axes[2]
xg = np.clip(frac, 1e-4, 1)
ax.scatter(xg[ug], r2[ug], s=5, c="#c1121f", alpha=0.35, lw=0, label="ungated")
ax.scatter(xg[gated], r2[gated], s=5, c=INK, alpha=0.35, lw=0, label="gated")
ax.set_xscale("log"); ax.axvline(MIN_PROP, color=MUTED, ls=(0, (4, 3)), lw=1)
ax.set_xlabel("the gate's statistic: largest share of bins above the patient's threshold (log)", fontsize=8); ax.set_ylabel("r2", fontsize=8)
ax.set_title("C  fit against the gate's own statistic (dashed: the 2 % rule)", fontsize=9, loc="left"); ax.legend(frameon=False, fontsize=7, loc="lower right")
ax = axes[3]
x = np.arange(K)
ax.bar(x - 0.28, PT.gated_sure, 0.26, color=[COL[j] for j in ORDER], label=f"gated members at loading >= {L}")
ax.bar(x, PT.ungated_sure, 0.26, color=[COL[j] for j in ORDER], alpha=0.45, hatch="////", label=f"ungated projected at >= {L}")
ax.bar(x + 0.28, PT.ungated_fit, 0.26, color=[COL[j] for j in ORDER], alpha=0.8, hatch="xx", label=f"of those, r2 >= {R2MIN}")
for xi, (g_, u_, w_) in enumerate(zip(PT.gated_sure, PT.ungated_sure, PT.ungated_fit)):
    for xx, v in ((xi - 0.28, g_), (xi, u_), (xi + 0.28, w_)):
        ax.text(xx, v + 3, str(v), ha="center", fontsize=6, color=INK)
ax.set_xticks(x); ax.set_xticklabels([f"c{j}" for j in ORDER], fontsize=8)
ax.set_title("D  members per type", fontsize=9, loc="left"); ax.legend(frameon=False, fontsize=7)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7)
fig.suptitle(f"The {int(ug.sum())} ungated electrodes projected onto the seven fixed archetypes (no refit) against the {int(gated.sum())} gated, cohort v17", x=0.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "fig1_projection.png", facecolor="white"); plt.close(fig)

# ---------------------------------------------------------------- 7. figure 2: per type, the two rows against the maps
mnames = [m[0] for m in MAPS]
labels_x = [m[0].replace("_vs_", " vs ").replace("_", " ") for m in MAPS]
norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2); cmap = plt.get_cmap("RdBu_r")
nr = 3 * K
fig, ax = plt.subplots(figsize=(0.62 * len(mnames) + 4.5, 0.42 * nr + 2.4), dpi=170)
E = np.full((nr, len(mnames)), np.nan)
for r_, (j, gname) in enumerate([(j, g) for j in ORDER for g in ("gated", "ungated", "wellfit")]):
    sub = MI[(MI.type == j) & (MI.group == gname)].set_index("map")
    y = nr - 1 - r_
    for c_, m in enumerate(mnames):
        q = sub.loc[m]
        e = float(q.enrichment) if np.isfinite(q.enrichment) and q.n >= 5 else np.nan
        E[r_, c_] = e
        if not np.isfinite(e):
            ax.text(c_, y, "-", ha="center", va="center", fontsize=6, color=MUTED); continue
        le = np.log2(max(e, 1e-3))
        ax.add_patch(plt.Rectangle((c_ - 0.5, y - 0.5), 1, 1, fc=cmap(norm(np.clip(le, -2, 2))), ec="none"))
        if q.outlined:
            ax.add_patch(plt.Rectangle((c_ - 0.46, y - 0.46), 0.92, 0.92, fill=False, ec=INK, lw=1.4))
        ax.text(c_, y, f"{q.pct_inside:.0f}", ha="center", va="center", fontsize=6.2, color="white" if abs(le) > 1.1 else INK)
    n_ = int(sub.n.iloc[0]); p_ = int(sub.patients.iloc[0])
    lab_ = f"c{j}  {names[j]}" if gname == "gated" else ("ungated, projected" if gname == "ungated" else f"ungated, r2 >= {R2MIN}")
    ax.text(-0.7, y, lab_ + f"   n {n_}, {p_} pat.",
            ha="right", va="center", fontsize=7, color=COL[j] if gname == "gated" else MUTED, fontweight="bold" if gname == "gated" else "normal")
for r_ in range(0, nr, 3):
    ax.axhline(nr - 1 - r_ - 2.5, color=GREY, lw=0.6)
# the baseline under each map and the stage over it
base = MI.groupby("map").baseline_pct.first()
ax.set_xticks(range(len(mnames))); ax.set_xticklabels([f"{l}{NL}{base[m]:.0f} %" for l, m in zip(labels_x, mnames)], rotation=90, fontsize=6.5)
stages = [m[1] for m in MAPS]
for s in dict.fromkeys(stages):
    cs = [i for i, x in enumerate(stages) if x == s]
    if s:
        ax.text(np.mean(cs), nr - 0.35, s, ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.plot([min(cs) - 0.4, max(cs) + 0.4], [nr - 0.42] * 2, color=INK, lw=0.6)
ax.set_xlim(-0.5, len(mnames) - 0.5); ax.set_ylim(-0.5, nr + 0.4); ax.set_yticks([])
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.tick_params(length=0)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, ticks=[-2, -1, 0, 1, 2]); cb.ax.set_yticklabels(["¼×", "½×", "1×", "2×", "4×"], fontsize=7)
cb.set_label("share inside against every seen electrode", fontsize=7)
ax.set_title(f"Where the types sit, gated members (loading >= {L}) and the ungated electrodes the projection gives them: % inside each map ({a.ns}), "
             f"colour = against all {len(keys)} seen electrodes, outlined = 95 % patient-bootstrap interval above 1, no spatial null; under each map its baseline",
             fontsize=7.5, loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig2_maps.png", facecolor="white"); plt.close(fig)
print("wrote", OUT)
