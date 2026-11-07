"""Step 16 of member_measures (Lora, 2026-11-05): one figure per chosen solution, every cluster, and only the maps that
at least one CLEAN cluster fits, at one minimum loading. Reads the cNMF runs directly and step 4's map table; renders
with VTK (desktop unlocked) or reuses step 15's tiles with --no-render.

    python 16_highlight_maps.py --ns nimare
    python 16_highlight_maps.py --ns nimare --configs 15band:7 15zband:7 5band:6 --loading 0.35 --perm 2000

    out/highlight_maps/<tag>_K<k>_load035.png          transparent background, with the glass brains
    out/highlight_maps/<tag>_K<k>_load035_noglass.png  the same without the glass brains
    out/highlight_maps/<tag>_K<k>_load035.csv     every cluster x every map (all 23), with q and the selection

THE SELECTION: for the solution, every cluster x all 23 maps is tested against the shaft-shift null (memberships
rotated along the shaft order within each patient, --perm rotations), Benjamini-Hochberg over all clusters x 23 maps.
A map can be shown when at least one CLEAN cluster (at most 50% of its members from one patient) fits it at q < 0.05;
at most --max-maps (10) are shown, shared out in turns: the clusters with the fewest fits choose first, and each takes
one map per round, its own maps first (fitted by no other clean cluster), then shared ones, largest difference first
(% of the cluster inside - % of all electrodes). A cluster with few fits keeps all of them; the language cluster, which
fits almost every map, fills the remaining places. The maps left out are
printed and marked in the CSV. The outlines use the same q, so the selection adds no test of its own.
LAYOUT: x = the clusters in the order of the cluster cards (the published 15-band types in P2.block_order: c5, c0,
c2, c6, c4, c1, c3); another solution's clusters take the place and the colour of the 15-band type they match (FIG 4's
rule, as the poster's z-scored companion; unmatched clusters last, in grey), ids are their own; in each cluster's
group one bar per selected map, always in the same order, numbered under the bars; under each group the cluster's
members on the glass brain, left hemisphere above right; at the bottom the selected maps on the pial surface in the
same order and with the same numbers, left above right, coloured where the map is in its top 10% (key on the left).
    bar        % of the cluster's members (own loading >= --loading) inside the map (the map's top 10%)
    black tick % of ALL electrodes inside the map
    error bar  95% from bootstrapping patients
    outline    the fit above, q < 0.05
    red dot    the cluster has more than 50% of its members from one patient
Map order: grouped by the first cluster (in x order) that fits it cleanly, largest difference first."""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as PE
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

import mm_common as MC

ap = argparse.ArgumentParser()
ap.add_argument("--configs", nargs="+", default=["15band:7", "15zband:7", "5band:6"], help="<tag>:<K>")
ap.add_argument("--loading", type=float, default=0.35)
ap.add_argument("--perm", type=int, default=2000)
ap.add_argument("--no-render", action="store_true")
ap.add_argument("--max-maps", type=int, default=10, help="at most this many maps per figure")
ap.add_argument("--all-maps", action="store_true", help="show all maps, not only the selected ones (files _all)")
CHOSEN = ["heard_sentences_vs_resting_state", "pictures_vs_resting_state", "reading_vs_resting_state",
          "production_vs_resting_state", "sentences_vs_words", "semantic_vs_resting_state"]   # production vs motor out (11-07)
ap.add_argument("--maps", nargs="*", default=None,
                help="show exactly these maps in this order (no names = the professor's four vs resting state plus sentences "
                     "vs words and semantic vs rest); the tests and BH still run over every map (files _sel)")
ap.add_argument("--v2", action="store_true",
                help="draw each bar's shaft-shift null (mean and 2.5-97.5 pct), fits solid with a 2.5 pt edge, the rest at "
                     "35% opacity; 300 dpi on white; files get _v2 (plotting only, every statistic unchanged)")
ap.add_argument("--v4", action="store_true",
                help="as --v3 plus the chance range (2.5-97.5 % of the shifts) as a grey box on each bar, 'above chance' decided "
                     "by the z-based p (files _v4)")
ap.add_argument("--v3", action="store_true",
                help="as --v2, without the patient-bootstrap whisker and the null's 95 % whisker (files _v3)")
MC.add_style_arg(ap)
a, _ = ap.parse_known_args()
if a.v4:
    a.v3 = True
if a.v3:
    a.v2 = True
if a.maps is not None and not a.maps:
    a.maps = CHOSEN
VSUF = ("_sel" if a.maps else "_all" if a.all_maps else "") + ("_v4" if a.v4 else ("_v3" if a.v3 else ("_v2" if a.v2 else "")))

FSET = {t: f for f, t in MC.TAG.items()}
RED = "#d7191c"
ONE_PATIENT = 0.5
LTAG = f"load{int(round(100 * a.loading)):03d}"
P2 = MC.p2()
OUTD = MC.SHARED / "highlight_maps"
OUTD.mkdir(parents=True, exist_ok=True)
TILES = MC.SHARED / "cross_feature_maps" / "tiles"                   # shared with step 15
TILES.mkdir(parents=True, exist_ok=True)
NIM = MC.SHARE.parent / "04_FBM_Pooling" / "neurosynth_nimare"
A305 = np.array([[0.9975, -0.0073, 0.0176, -0.0429], [0.0146, 1.0009, -0.0024, 1.5496],
                 [-0.0130, -0.0093, 0.9971, 1.1840], [0.0, 0.0, 0.0, 1.0]])
NS_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("ns", ["#f7b267", "#d7301f", "#7f0000"])
ns = pd.read_csv(MC.NS_Z).drop(columns="type", errors="ignore").set_index("key")
order = pd.read_csv(MC.NS_MAPS).column.tolist() if MC.NS_MAPS.exists() else []
MAPS = [c for c in order if c in ns.columns] + [c for c in ns.columns if c.startswith("ns_") and c not in order]
NAMES = [m[3:].replace("_vs_", " vs ").replace("_", " ") for m in MAPS]
P_LANA = pd.read_csv(MC.SHARE / "member_measures" / "out_k7" / "15band" / "measures.csv").set_index("key").P_lana
rng = np.random.default_rng(0)


def shaft_key(name):
    r = re.match(r"^(.*?)(\d+)$", str(name))
    return (r.group(1), int(r.group(2))) if r else (str(name), 0)


def glass_tile(tag, K, j, side, d, colour):
    p = TILES / f"{tag}_K{K}_c{j}_{side}_{LTAG}_{''.join(f'{int(255 * c):02x}' for c in colour[:3])}.png"
    if p.exists() and a.no_render:
        return np.asarray(Image.open(p).convert("RGBA"))
    if a.no_render:
        raise SystemExit(f"--no-render but no tile {p.name}: run once with the desktop unlocked")
    d2 = dict(d)
    G = d["Gn"].copy()
    G[d["lab"] != j, j] = 0.0                                        # members only, as in the bars
    d2["Gn"] = G
    orig_cut, orig_col = P2.LOAD_CUT, P2.cluster_col
    P2.LOAD_CUT = a.loading
    P2.cluster_col = lambda jj, kk: colour
    try:
        img, _ = P2.render_hemi(side, d2, j)
    finally:
        P2.LOAD_CUT, P2.cluster_col = orig_cut, orig_col
    Image.fromarray(img).save(p)
    return img


def pial_tile(name, side):
    """The map on the pial surface where it is in its top 10% (as in step 15)."""
    p = TILES / f"map_{name}_{side}.png"
    if p.exists():
        return np.asarray(Image.open(p).convert("RGBA"))
    if a.no_render:
        raise SystemExit(f"--no-render but no tile {p.name}: run once with the desktop unlocked")
    import nibabel as nb
    img = nb.load(str(NIM / name / f"{name}_association-test_z.nii.gz"))
    vol = np.asarray(img.dataobj, float)
    inmask = vol[vol != 0]
    thr, top = np.quantile(inmask, 0.90), np.quantile(inmask, 0.995)
    inv = np.linalg.inv(img.affine)
    m = P2.hemi_mesh(side).copy()
    v, nrm = np.asarray(m.points, float), np.asarray(m.point_normals, float)
    best = np.full(len(v), -np.inf)
    for off in (0.0, 1.5, -1.5, 3.0, -3.0):
        xyz = v + off * nrm
        mni = (np.c_[xyz, np.ones(len(xyz))] @ A305.T)[:, :3]
        ijk = np.round((np.c_[mni, np.ones(len(mni))] @ inv.T)[:, :3]).astype(int)
        ok = (ijk >= 0).all(1) & (ijk < np.array(vol.shape[:3])).all(1)
        z = np.full(len(v), -np.inf)
        z[ok] = vol[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        best = np.maximum(best, z)
    rgb = np.tile(np.array(matplotlib.colors.to_rgb(P2.RC.BRAIN_COLOR)) * 255, (len(v), 1))
    ins = best >= thr
    rgb[ins] = NS_CMAP(np.clip((best[ins] - thr) / max(top - thr, 1e-6), 0, 1))[:, :3] * 255
    m["rgb"] = rgb.astype(np.uint8)
    pl = P2.pv.Plotter(off_screen=True, window_size=P2.RENDER_PX)
    pl.add_mesh(m, scalars="rgb", rgb=True, smooth_shading=True, specular=P2.RC.BRAIN_SPECULAR,
                specular_power=P2.RC.BRAIN_SPECULAR_POWER, ambient=P2.RC.BRAIN_AMBIENT, diffuse=P2.RC.BRAIN_DIFFUSE)
    b = m.bounds
    cx, cy, cz = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2
    dist = 2.4 * max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
    pl.camera_position = ((cx - dist if side == "L" else cx + dist, cy, cz), (cx, cy, cz), (0, 0, 1))
    pl.enable_parallel_projection()
    W, H = P2.RENDER_PX
    vis = 2.0 * pl.camera.parallel_scale
    pl.camera.zoom(min(1.30, min(vis / (1.06 * (b[5] - b[4])), vis * (W / H) / (1.06 * (b[3] - b[2])))))
    out = P2._crop_alpha(np.asarray(pl.screenshot(return_img=True, transparent_background=True)))
    pl.close()
    Image.fromarray(out).save(p)
    return out


def show(fig, img, x0, y0, w, h, W, H):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    ww, hh = iw * s, ih * s
    ax = fig.add_axes([(x0 + (w - ww) / 2) / W, (y0 + (h - hh) / 2) / H, ww / W, hh / H])
    ax.imshow(img)
    ax.axis("off")


# the cluster cards' order: the published 15-band types at K 7 in P2.block_order, and FIG 4's matching to them
F4 = MC.load_module(MC.SHARE / "00_paper2_figure4_correspondence.py", "mm_f4")
REF = P2.load_run("concat_bands15", 7)
_C = P2.cube(REF["X"], REF)
REF_KEYS = pd.Index([f"{p}|{P2.norm(e)}" for p, e in zip(REF["meta"]["patient_id"], REF["meta"]["electrode"])])
CARD = [int(j) for j in P2.block_order(np.stack([_C[REF["lab"] == j].mean(0) for j in range(7)]), REF)[0]]
SPACE = F4.description_space()
ZZ = F4.electrode_z(SPACE["Z"])
_sref = F4.solution("cnmf", "concat_bands15", 7)
F4.align(SPACE, _sref)
C_REF = F4.centroids(ZZ, _sref["W"])
GREY = (0.60, 0.60, 0.60)


def card_types(tag, K):
    """{cluster: the 15-band K 7 type it stands for, or None}: itself for 15 bands K 7, else FIG 4's greedy match."""
    if tag == "15band" and K == 7:
        return {j: j for j in range(K)}
    s_ = F4.solution("cnmf", FSET[tag], K)
    F4.align(SPACE, s_)
    R = F4.corr_matrix(C_REF, F4.centroids(ZZ, s_["W"]), centre=True)
    m = {j: i for i, j, r in F4.greedy_match(R)}
    return {j: m.get(j) for j in range(K)}


for cfg in a.configs:
    tag, K = cfg.split(":")[0], int(cfg.split(":")[1])
    rng = np.random.default_rng(0)                                     # a fresh stream per configuration (= steps 13, 15)
    d = P2.load_run(FSET[tag], K)
    keys = [f"{p}|{P2.norm(e)}" for p, e in zip(d["meta"]["patient_id"], d["meta"]["electrode"])]
    Zm = ns.reindex(keys)[MAPS].to_numpy(float)
    has = np.isfinite(Zm).all(1)
    I = (Zm > 0).astype(float)
    lab = d["lab"]
    load = d["Gn"][np.arange(len(lab)), lab]
    pat = d["patient"]
    sk = [shaft_key(e) for e in d["meta"]["electrode"].to_numpy()]
    coh = 100 * I[has].mean(0)
    Pl = P_LANA.reindex(keys).to_numpy(float)
    TYPE = card_types(tag, K)
    rank = sorted(range(K), key=lambda j: (TYPE[j] is None, CARD.index(TYPE[j]) if TYPE[j] is not None else j))
    COL = {j: (tuple(np.asarray(P2.cluster_col(TYPE[j], 7))[:3]) if TYPE[j] is not None else GREY) for j in range(K)}
    kp = np.where(has & (load >= a.loading))[0]
    lk, pk, Ik = lab[kp], pat[kp], I[kp]
    pats = np.unique(pk)
    pidx = pd.Series(pk).map({p: i for i, p in enumerate(pats)}).to_numpy()
    groups = [g for g in (np.array(sorted(np.where(pidx == q)[0], key=lambda i: sk[kp[i]])) for q in range(len(pats)))
              if len(g) > 1]
    share = lambda lb: np.array([Ik[lb == j].mean(0) if (lb == j).any() else np.zeros(len(MAPS)) for j in range(K)])
    obs = share(lk)
    ge = np.zeros_like(obs)
    nulls = np.empty((a.perm,) + obs.shape)                            # every rotation's share (for the figure only)
    for i_ in range(a.perm):
        perm = np.arange(len(kp))
        for g in groups:
            perm[g] = np.roll(g, rng.integers(1, len(g)))
        nulls[i_] = share(lk[perm])
        ge += nulls[i_] >= obs                                         # one-sided: rotations at least as high as observed
    NMEAN = 100 * nulls.mean(0)
    NLO, NHI = 100 * np.quantile(nulls, [0.025, 0.975], axis=0)
    # the professor's z: how many null SDs the real share sits above the null mean, and the normal-tail p of that z
    # (no floor at 1/(perm+1)); its BH q over the same K x 23 cells; the null's skewness, since the normal tail is only
    # trustworthy where the shifts are roughly bell-shaped (|skew| > 1 flagged)
    from scipy.stats import norm, skew as _skew
    NSD = 100 * nulls.std(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ZS = np.where(NSD > 0, (100 * obs - NMEAN) / NSD, np.nan)
    PZ = norm.sf(ZS)
    _f = np.isfinite(PZ).ravel()
    _pz = PZ.ravel()[_f]
    _o = np.argsort(_pz)
    _q = np.minimum.accumulate((_pz[_o] * len(_pz) / (np.arange(len(_pz)) + 1))[::-1])[::-1]
    QZ = np.full(PZ.size, np.nan)
    _qq = np.empty_like(_q)
    _qq[_o] = np.minimum(_q, 1)
    QZ[np.where(_f)[0]] = _qq
    QZ = QZ.reshape(PZ.shape)
    SKEW = _skew(nulls, axis=0)
    # two-sided: a cluster can sit above OR below chance; p = 2 x the normal tail beyond |z|, BH over all cells
    P2Z = 2 * norm.sf(np.abs(ZS))
    _f2 = np.isfinite(P2Z).ravel()
    _p2 = P2Z.ravel()[_f2]
    _o2 = np.argsort(_p2)
    _q2 = np.minimum.accumulate((_p2[_o2] * len(_p2) / (np.arange(len(_p2)) + 1))[::-1])[::-1]
    Q2Z = np.full(P2Z.size, np.nan)
    _qq2 = np.empty_like(_q2)
    _qq2[_o2] = np.minimum(_q2, 1)
    Q2Z[np.where(_f2)[0]] = _qq2
    Q2Z = Q2Z.reshape(P2Z.shape)
    pv = ((ge + 1) / (a.perm + 1)).ravel()
    o = np.argsort(pv)
    qv = np.minimum.accumulate((pv[o] * len(pv) / (np.arange(len(pv)) + 1))[::-1])[::-1]
    Q = np.empty_like(qv)
    Q[o] = np.minimum(qv, 1)
    Q = Q.reshape(obs.shape)                                          # K x 23, BH over all of them
    n_ = np.array([(lk == j).sum() for j in range(K)])
    top = np.array([pd.Series(pk[lk == j]).value_counts().iloc[0] / n_[j] if n_[j] else 1.0 for j in range(K)])
    clean = top <= ONE_PATIENT
    B = rng.multinomial(len(pats), np.full(len(pats), 1 / len(pats)), size=1000).astype(float)
    LO, HI = np.full((K, len(MAPS)), np.nan), np.full((K, len(MAPS)), np.nan)
    for j in range(K):
        sel = lk == j
        if not sel.any():
            continue
        Sm = np.zeros((len(pats), len(MAPS)))
        np.add.at(Sm, pidx[sel], Ik[sel])
        Nm = np.bincount(pidx[sel], minlength=len(pats)).astype(float)
        den = B @ Nm
        LO[j], HI[j] = 100 * np.quantile((B @ Sm)[den > 0] / den[den > 0, None], [0.025, 0.975], axis=0)
    # --v4 decides with the z-based p (the professor's: no floor at 1/(perm+1)), the others with the counting p
    fit = ((ZS > 0) & (Q2Z < 0.05)) if a.v4 else (Q < 0.05)
    below = (ZS < 0) & (Q2Z < 0.05)                                    # drawn only in --v4
    # the corrected chance box (--v4): BH at q < 0.05 rejects exactly the cells whose two-sided p is <= the largest p
    # that passes, so |z| >= z_crit with z_crit = the normal quantile of that p; the box null mean +/- z_crit x null SD
    # then holds every bar that is NOT significant and none that is (no cell passing: the first BH step, 0.05 / cells)
    _ok = np.isfinite(P2Z) & (Q2Z < 0.05)
    P_CUT = float(P2Z[_ok].max()) if _ok.any() else 0.05 / np.isfinite(P2Z).sum()
    Z_CRIT = float(norm.isf(P_CUT / 2))
    CLO, CHI = np.maximum(NMEAN - Z_CRIT * NSD, 0.0), NMEAN + Z_CRIT * NSD
    _out_hi, _out_lo = 100 * obs >= CHI - 1e-9, 100 * obs <= NMEAN - Z_CRIT * NSD + 1e-9  # the cut-off bar sits ON the edge
    _bad = int(((_out_hi != fit) | (_out_lo != below))[np.isfinite(ZS)].sum()) if a.v4 else 0
    print(f"  corrected box: BH cut-off p {P_CUT:.4g} -> |z| >= {Z_CRIT:.2f}; bars outside the box that are not "
          f"significant, or inside that are: {_bad}")
    # the selection: maps fitted by at least one clean cluster, at most --max-maps of them, shared out between the
    # clusters in turns: the clusters with the fewest fits choose first, each takes its next map per round - its own
    # maps first (fitted by no other clean cluster), then shared ones - largest difference first (% of the cluster
    # inside - % of all electrodes). A cluster with few fits therefore keeps all of them, and the language cluster,
    # which fits almost every map, fills the rest.
    diff = 100 * obs - coh[None, :]
    fitters = {m: [j for j in range(K) if clean[j] and fit[j, m]] for m in range(len(MAPS))}
    queue = {j: sorted([m for m in range(len(MAPS)) if j in fitters[m]],
                       key=lambda m: (len(fitters[m]) > 1, -diff[j, m])) for j in range(K) if clean[j] and fit[j].any()}
    sel = []
    while len(sel) < a.max_maps and any(queue.values()):
        for j in sorted(queue, key=lambda j: (int(fit[j].sum()), rank.index(j))):
            while queue[j] and queue[j][0] in sel:
                queue[j].pop(0)
            if queue[j] and len(sel) < a.max_maps:
                sel.append(queue[j].pop(0))
    if a.all_maps:                                                    # every map, fitted ones first
        sel = sel + [m for m in range(len(MAPS)) if m not in sel and fitters[m]] + [m for m in range(len(MAPS)) if not fitters[m]]
    if a.maps:                                                        # exactly these, in this order
        missing = [n for n in a.maps if "ns_" + n not in MAPS]
        if missing:
            raise SystemExit(f"--maps: not in the z table {missing} (run 0_nimare_maps.py and 4_neurosynth.py)")
        sel = [MAPS.index("ns_" + n) for n in a.maps]
    dropped = [MAPS[m][3:] for m, js in fitters.items() if js and m not in sel]
    chosen = []                                                       # display order: by the first fitting cluster in x
    for j in rank:
        mine = [m for m in sel if m not in chosen and j in fitters[m]]
        chosen += sorted(mine, key=lambda m: -diff[j, m])
    chosen += [m for m in sel if m not in chosen]                     # maps no clean cluster fits (--all-maps)
    if a.maps:
        chosen = list(sel)
    rows = [dict(fset=tag, k=K, loading=a.loading, cluster=j, n=int(n_[j]), top_patient_share=round(top[j], 2),
                 clean=bool(clean[j]), map=MAPS[m][3:], pct_inside=round(100 * obs[j, m], 1),
                 lo=round(LO[j, m], 1), hi=round(HI[j, m], 1), cohort_pct=round(coh[m], 1),
                 null_mean=round(NMEAN[j, m], 1), null_lo=round(NLO[j, m], 1), null_hi=round(NHI[j, m], 1),
                 p_shaft=round(float(pv.reshape(obs.shape)[j, m]), 4), q_shaft=round(Q[j, m], 4),
                 null_sd=round(NSD[j, m], 2), z=round(float(ZS[j, m]), 2), p_z=float(f"{PZ[j, m]:.3g}"),
                 q_z=float(f"{QZ[j, m]:.3g}"), null_skew=round(float(SKEW[j, m]), 2),
                 skewed_null=bool(abs(SKEW[j, m]) > 1),
                 fit=bool(fit[j, m]), fit_rule="two-sided z-based q" if a.v4 else "counting q",
                 p_z_two_sided=float(f"{P2Z[j, m]:.3g}"), q_z_two_sided=float(f"{Q2Z[j, m]:.3g}"),
                 below_chance=bool(below[j, m]), z_crit=round(Z_CRIT, 3),
                 box_lo_corrected=round(float(CLO[j, m]), 2), box_hi_corrected=round(float(CHI[j, m]), 2),
                 fit_counting=bool(Q[j, m] < 0.05), fit_z=bool(QZ[j, m] < 0.05), shown=m in chosen) for j in range(K) for m in range(len(MAPS))]
    pd.DataFrame(rows).to_csv(OUTD / f"{tag}_K{K}_{LTAG}{VSUF}.csv", index=False)
    if not chosen:
        print(f"  {tag} K {K}: no map is fitted by a clean cluster, no figure")
        continue

    # ---- the figure, in inches, with and without the glass brains
    for GLASS in (True, False):
        M = len(chosen)
        BARW = 0.2
        GW = M * BARW + 0.45                                              # one cluster's group
        LEFT, RIGHT = 1.0, 0.4
        PLOT_W = K * GW
        W = LEFT + PLOT_W + RIGHT
        MAPW = PLOT_W / M                                                 # one map's pial column
        PIAL_H = min(0.9, MAPW * 0.62)
        GLASS_H = min(1.15, (GW - 0.1) * 0.62) if GLASS else 0.0
        # --v4: the centroid beside the brains when the group has room (all maps), else in its own row above them
        CENT = GLASS and a.v4
        SIDE = CENT and (GW - 0.1) >= GLASS_H * 1.38 + 2.9
        CENT_H = (GW - 0.1) / 2.4 + 0.3 if CENT and not SIDE else 0.0
        H_BAR, TOPM = 3.0, 0.75
        Y_PIAL = 0.55
        Y_GLASS = Y_PIAL + 2 * PIAL_H + (0.75 if GLASS else 0.55)
        Y_BAR = Y_GLASS + 2 * GLASS_H + CENT_H + 0.55
        H = Y_BAR + H_BAR + TOPM
        fig = plt.figure(figsize=(W, H), dpi=300 if a.v2 else 200)
        fig.patch.set_alpha(0)
        ax = fig.add_axes([LEFT / W, Y_BAR / H, PLOT_W / W, H_BAR / H])
        ax.patch.set_alpha(0)
        ymax = (max(np.nanmax(100 * obs[:, chosen]), np.nanmax(CHI[:, chosen]), 10) * 1.08 if a.v4   # bars and chance boxes only
                else max(np.nanmax(HI[:, chosen]), 10) * 1.05)
        xt, xl = [], []
        for xi, j in enumerate(rank):
            col = np.asarray(COL[j])
            g0 = xi * GW + 0.225
            for k_, m in enumerate(chosen):
                x = g0 + (k_ + 0.5) * BARW
                if a.v2:
                    # fits solid with a 2.5 pt black edge, the rest the same hue at 35 % opacity; patient-bootstrap
                    # interval grey on the left half, the shaft-shift null (2.5-97.5 pct whisker + mean tick) black on
                    # the right half, all electrodes as a dashed line across the bar
                    if a.v4:                                        # opaque: full colour / pale tint / white with a coloured edge
                        face = col if fit[j, m] else ((1, 1, 1) if below[j, m] else 0.38 * col + 0.62)
                        ax.bar(x, 100 * obs[j, m], width=BARW * 0.58, color=face, zorder=2,
                               edgecolor=tuple(0.75 * col) if below[j, m] else "none", lw=1.1 if below[j, m] else 0)
                    else:
                        ax.bar(x, 100 * obs[j, m], width=BARW * 0.88, color=col, alpha=1.0 if fit[j, m] else 0.35,
                               edgecolor="black" if fit[j, m] else "none", lw=2.5 if fit[j, m] else 0, zorder=2)
                    xb, xn = x - BARW * 0.2, x + BARW * 0.2
                    if not a.v3:
                        ax.plot([xb, xb], [LO[j, m], HI[j, m]], color=P2.MUTED, lw=0.7, zorder=3)
                        ax.plot([xn, xn], [NLO[j, m], NHI[j, m]], color="black", lw=0.9, zorder=4)
                    else:
                        xn = x                                          # the null tick centred on the bar
                        if a.v4:                                        # the corrected chance box, wider than the bar, behind it
                            ax.add_patch(matplotlib.patches.Rectangle((x - BARW * 0.43, CLO[j, m]), BARW * 0.86,
                                                                      CHI[j, m] - CLO[j, m], facecolor="#e3e5e8",
                                                                      edgecolor="none", zorder=1))
                            if below[j, m]:                             # below chance: a small down triangle under the box
                                ax.plot(x, CLO[j, m], marker="v", color="#2b3038", ms=5, mew=0, zorder=5)
                    hw = BARW * (0.43 if a.v4 else 0.17)                # --v4: the chance line spans the whole box
                    ax.plot([xn - hw, xn + hw], [NMEAN[j, m]] * 2, color="#2b3038" if a.v4 else "black",
                            lw=1.3 if a.v4 else 1.8, solid_capstyle="butt", zorder=4)
                    if not a.v3:
                        ax.plot([x - BARW * 0.46, x + BARW * 0.46], [coh[m]] * 2, color="black", lw=0.9, ls=(0, (2, 1.5)), zorder=4)
                else:
                    ax.bar(x, 100 * obs[j, m], width=BARW * 0.88, color=col, edgecolor=P2.INK if fit[j, m] else "none",
                           lw=1.1 if fit[j, m] else 0, zorder=2)
                    ax.plot([x, x], [LO[j, m], HI[j, m]], color=P2.INK, lw=0.5, zorder=3)
                    ax.plot([x - BARW * 0.46, x + BARW * 0.46], [coh[m]] * 2, color="black", lw=1.0, zorder=4)
                if not clean[j]:
                    ax.plot(x, min(HI[j, m] + 3, ymax * 0.98), "o", color=RED, ms=2.6, zorder=5)
                xt.append(x)
                xl.append(str(k_ + 1))
            if xi:
                ax.axvline(xi * GW, color="#e3e6e4", lw=0.6, zorder=0)
            # the cluster's id and n under its group, then its glass brains (left above right)
            xc = LEFT + g0 + M * BARW / 2
            fig.text(xc / W, (Y_GLASS + 2 * GLASS_H + CENT_H + 0.12) / H, f"c{j}   n {n_[j]}" + ("" if clean[j] else f"   {100 * top[j]:.0f}% one patient"),
                     ha="center", va="bottom", fontsize=9, fontweight="bold", color=RED if not clean[j] else col)
            gx0, gw_ = xc - (GW - 0.1) / 2, GW - 0.1
            if CENT:
                # the cluster's centroid: mean ERSP of the electrodes in its bars, on the 15-band cube, as on the cards
                ridx = REF_KEYS.get_indexer(keys)
                assert (ridx >= 0).all(), "electrodes missing from the 15-band run"
                Mj = _C[ridx[kp[lk == j]]].mean(0)
                if SIDE:
                    ew, eh = 2.6, 1.45
                    ex0, ey0 = gx0, Y_GLASS + GLASS_H - eh / 2 + 0.08
                    gx0, gw_ = gx0 + ew + 0.3, gw_ - ew - 0.3
                else:
                    ew, eh = gw_, CENT_H - 0.3
                    ex0, ey0 = gx0, Y_GLASS + 2 * GLASS_H + 0.15
                axc = fig.add_axes([(ex0 + 0.32) / W, (ey0 + 0.14) / H, (ew - 0.32) / W, (eh - 0.14) / H])
                P2.draw_mean(axc, Mj, REF, vlim=P2.MEAN_VLIM, col=tuple(col))
                nt = REF["nt"]
                axc.set_xticks([(i + 0.5) * nt for i in range(len(REF["conds"]))])
                axc.set_xticklabels([P2.COND_LABEL.get(c, c).split(" ")[0] for c in REF["conds"]], fontsize=6)
                axc.set_yticks([0, 4, 7, 9, 14])
                axc.set_yticklabels([REF["bands"][i].replace("Hz", "") for i in (0, 4, 7, 9, 14)], fontsize=5.5)
                axc.tick_params(length=0, pad=1.5, colors=P2.MUTED)
                for s_ in axc.spines.values():
                    s_.set_edgecolor(col)
                    s_.set_linewidth(2.0)
            for k_, side in enumerate(("L", "R") if GLASS else ()):
                show(fig, glass_tile(tag, K, j, side, d, tuple(col)), gx0, Y_GLASS + (1 - k_) * GLASS_H,
                     gw_, GLASS_H, W, H)
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=5.5, color="#4a5462")
        ax.tick_params(axis="x", length=0, pad=1.5)
        ax.set_xlim(0, PLOT_W)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("% of the cluster's electrodes inside the map", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{MC.LABEL[tag]}, K = {K}, members with loading ≥ {a.loading:g}: "
                     + (f"{M} chosen maps" if a.maps else f"all {M} maps" if a.all_maps else f"the {M} maps that at least one clean cluster fits"),
                     fontsize=10.5, loc="left")
        _bl = np.array([0.22, 0.56, 0.82])
        V4H = [Patch(facecolor=_bl, edgecolor="none", label="above chance"),
               Patch(facecolor=0.38 * _bl + 0.62, edgecolor="none", label="neither"),
               Patch(facecolor="white", edgecolor=0.75 * _bl, lw=1.1, label="below chance (avoids the map)"),
               Line2D([0], [0], color="#2b3038", lw=1.3, label="chance level (shaft-shift null)"),
               Patch(facecolor="#e3e5e8", edgecolor="none", label="chance range (corrected): outside = significant")]
        if not all(clean[j] for j in rank):                            # a one-patient cluster is drawn: say what the dot is
            V4H.append(Line2D([0], [0], marker="o", ls="", color=RED, ms=4, label="> 50% one patient"))
        ax.legend(handles=V4H if a.v4 else ([(Patch(facecolor="#398fd0", edgecolor="none", label="above chance: full colour") if a.v4 else
                             Patch(facecolor="#9aa0a6", edgecolor="black", lw=2.5, label="above chance (q < 0.05)" if a.v3 else "fits (q < 0.05, shaft-shift null)")),
                            (Line2D([0], [0], color="black", lw=1.8, ls="none", marker="_", ms=9, mew=1.8, label="chance level (shaft-shift null)",
                                    path_effects=[PE.Stroke(linewidth=3.6, foreground="white"), PE.Normal()] if a.v4 else None)
                             if a.v3 else Line2D([0], [0], color="black", lw=0.9, marker="_", ms=9, mew=1.8,
                                                 label="null mean and 95 % range")),
                            Line2D([0], [0], color="black", lw=0.9, ls=(0, (2, 1.5)), label="all electrodes"),
                            Line2D([0], [0], marker="o", ls="", color=RED, ms=4, label="> 50 % one patient")]
                           [:2 if a.v3 else 4]
                           + ([Patch(facecolor="#398fd0", alpha=0.35, edgecolor="none", label="neither: faded"),
                               Patch(facecolor=(0.22, 0.56, 0.82, 0.35), hatch="////", edgecolor=(0.12, 0.31, 0.45),
                                     lw=0, label="below chance (avoids the map): hatched + ▼")]
                              if a.v4 else [])
                           + ([Patch(facecolor="#d5d8dc", edgecolor="#6b727c", lw=0.7,
                                     label="chance range, corrected: a bar outside it is significant")] if a.v4 else [])
                           + ([Line2D([0], [0], marker="o", ls="", color=RED, ms=4, label="> 50 % one patient")]
                              if a.v3 and not all(clean[j] for j in rank) else []) if a.v2 else
                           [Patch(facecolor="white", edgecolor=P2.INK, label="fits: beats the shaft-shift null, q < 0.05"),
                            Line2D([0], [0], color="black", lw=1.2, label="all electrodes"),
                            Line2D([0], [0], marker="o", ls="", color=RED, ms=4, label="> 50% one patient")]),
                  frameon=False, fontsize=8 if a.v3 else 7.5, ncol=2 if a.v4 else (4 if a.v2 else 3),
                  **(dict(loc="upper left") if a.v4 else
                     dict(loc="lower right", bbox_to_anchor=(1.0, 1.0)) if a.v2 else dict(loc="upper right")))
        # the selected maps on the pial surface, same order and numbers, left above right
        for k_, m in enumerate(chosen):
            xc = LEFT + (k_ + 0.5) * MAPW
            for s_, side in enumerate(("L", "R")):
                show(fig, pial_tile(MAPS[m][3:], side), xc - MAPW * 0.47, Y_PIAL + (1 - s_) * PIAL_H, MAPW * 0.94, PIAL_H, W, H)
            fig.text(xc / W, (Y_PIAL - 0.05) / H, f"{k_ + 1}  " + NAMES[m].replace(" vs ", "\nvs "), ha="center", va="top",
                     fontsize=6.8, color="#1c2330", linespacing=1.1)
        if CENT:                                                        # the centroids' colour scale
            eax = fig.add_axes([0.25 / W, (Y_GLASS + GLASS_H) / H, 0.55 / W, 0.1 / H])
            eb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-P2.MEAN_VLIM, P2.MEAN_VLIM), cmap=P2.ERSP_CMAP),
                              cax=eax, orientation="horizontal", ticks=[-P2.MEAN_VLIM, 0, P2.MEAN_VLIM])
            eb.ax.set_xticklabels([f"-{P2.MEAN_VLIM:g}", "0", f"+{P2.MEAN_VLIM:g}"], fontsize=6)
            eb.ax.tick_params(length=2)
            eax.set_title("centroid\n(dB)", fontsize=6.5, pad=2)
        cax = fig.add_axes([0.25 / W, (Y_PIAL + PIAL_H * 0.6) / H, 0.55 / W, 0.1 / H])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=NS_CMAP), cax=cax, orientation="horizontal",
                          ticks=[0, 1])
        cb.ax.set_xticklabels(["90th\npct", "99.5th\npct+"], fontsize=6)
        cb.ax.tick_params(length=2)
        cax.set_title("coloured =\ntop 10%", fontsize=6.5, pad=2)
        out = OUTD / f"{tag}_K{K}_{LTAG}{'' if GLASS else '_noglass'}{VSUF}.png"
        if a.v2:
            fig.patch.set_facecolor("white")
            fig.patch.set_alpha(1.0)                                    # a white page, not a transparent one
            MC.save_styles(fig, out, a.styles, facecolor="white")
        else:
            MC.save_styles(fig, out, a.styles, transparent=True)
        plt.close(fig)
        print(f"wrote {out.name} ({W:.0f} x {H:.0f} in): {M} maps shown; fits per cluster: "
              + ", ".join(f"c{j}{'' if clean[j] else '(X)'} {int(fit[j].sum())}" for j in rank)
              + (f"; left out: {', '.join(dropped)}" if dropped else ""))
