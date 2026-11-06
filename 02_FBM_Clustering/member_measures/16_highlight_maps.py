"""Step 16 of member_measures (Lora, 2026-11-05): one figure per chosen solution, every cluster, and only the maps that
at least one CLEAN cluster fits, at one minimum loading. Reads the cNMF runs directly and step 4's map table; renders
with VTK (desktop unlocked) or reuses step 15's tiles with --no-render.

    python 16_highlight_maps.py --ns nimare
    python 16_highlight_maps.py --ns nimare --configs 15band:7 15zband:7 5band:6 --loading 0.35 --perm 2000

    out/highlight_maps/<tag>_K<k>_load035.png     transparent background
    out/highlight_maps/<tag>_K<k>_load035.csv     every cluster x every map (all 23), with q and the selection

THE SELECTION: for the solution, every cluster x all 23 maps is tested against the shaft-shift null (memberships
rotated along the shaft order within each patient, --perm rotations), Benjamini-Hochberg over all clusters x 23 maps.
A map can be shown when at least one CLEAN cluster (at most 50% of its members from one patient) fits it at q < 0.05;
at most --max-maps (10) are shown, shared out in turns: the clusters with the fewest fits choose first, and each takes
one map per round, its own maps first (fitted by no other clean cluster), then shared ones, largest difference first
(% of the cluster inside - % of all electrodes). A cluster with few fits keeps all of them; the language cluster, which
fits almost every map, fills the remaining places. The maps left out are
printed and marked in the CSV. The outlines use the same q, so the selection adds no test of its own.
LAYOUT: x = the clusters (order of mean P(LanA), the violins' order, ids in the cluster's colour); in each cluster's
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
a, _ = ap.parse_known_args()

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


for cfg in a.configs:
    tag, K = cfg.split(":")[0], int(cfg.split(":")[1])
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
    rank = sorted(range(K), key=lambda j: -np.nanmean(Pl[lab == j]))
    kp = np.where(has & (load >= a.loading))[0]
    lk, pk, Ik = lab[kp], pat[kp], I[kp]
    pats = np.unique(pk)
    pidx = pd.Series(pk).map({p: i for i, p in enumerate(pats)}).to_numpy()
    groups = [g for g in (np.array(sorted(np.where(pidx == q)[0], key=lambda i: sk[kp[i]])) for q in range(len(pats)))
              if len(g) > 1]
    share = lambda lb: np.array([Ik[lb == j].mean(0) if (lb == j).any() else np.zeros(len(MAPS)) for j in range(K)])
    obs = share(lk)
    ge = np.zeros_like(obs)
    for _ in range(a.perm):
        perm = np.arange(len(kp))
        for g in groups:
            perm[g] = np.roll(g, rng.integers(1, len(g)))
        ge += share(lk[perm]) >= obs
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
    fit = Q < 0.05
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
    dropped = [MAPS[m][3:] for m, js in fitters.items() if js and m not in sel]
    chosen = []                                                       # display order: by the first fitting cluster in x
    for j in rank:
        mine = [m for m in sel if m not in chosen and j in fitters[m]]
        chosen += sorted(mine, key=lambda m: -diff[j, m])
    rows = [dict(fset=tag, k=K, loading=a.loading, cluster=j, n=int(n_[j]), top_patient_share=round(top[j], 2),
                 clean=bool(clean[j]), map=MAPS[m][3:], pct_inside=round(100 * obs[j, m], 1),
                 lo=round(LO[j, m], 1), hi=round(HI[j, m], 1), cohort_pct=round(coh[m], 1), q_shaft=round(Q[j, m], 4),
                 fit=bool(fit[j, m]), shown=m in chosen) for j in range(K) for m in range(len(MAPS))]
    pd.DataFrame(rows).to_csv(OUTD / f"{tag}_K{K}_{LTAG}.csv", index=False)
    if not chosen:
        print(f"  {tag} K {K}: no map is fitted by a clean cluster, no figure")
        continue

    # ---- the figure, in inches
    M = len(chosen)
    BARW = 0.2
    GW = M * BARW + 0.45                                              # one cluster's group
    LEFT, RIGHT = 1.0, 0.4
    PLOT_W = K * GW
    W = LEFT + PLOT_W + RIGHT
    MAPW = PLOT_W / M                                                 # one map's pial column
    PIAL_H = min(0.9, MAPW * 0.62)
    GLASS_H = min(1.15, (GW - 0.1) * 0.62)
    H_BAR, TOPM = 3.0, 0.75
    Y_PIAL = 0.55
    Y_GLASS = Y_PIAL + 2 * PIAL_H + 0.75
    Y_BAR = Y_GLASS + 2 * GLASS_H + 0.55
    H = Y_BAR + H_BAR + TOPM
    fig = plt.figure(figsize=(W, H), dpi=200)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([LEFT / W, Y_BAR / H, PLOT_W / W, H_BAR / H])
    ax.patch.set_alpha(0)
    ymax = max(np.nanmax(HI[:, chosen]), 10) * 1.05
    xt, xl = [], []
    for xi, j in enumerate(rank):
        col = np.asarray(P2.cluster_col(j, K))[:3]
        g0 = xi * GW + 0.225
        for k_, m in enumerate(chosen):
            x = g0 + (k_ + 0.5) * BARW
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
        fig.text(xc / W, (Y_GLASS + 2 * GLASS_H + 0.12) / H, f"c{j}   n {n_[j]}" + ("" if clean[j] else f"   {100 * top[j]:.0f}% one patient"),
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color=RED if not clean[j] else col)
        for k_, side in enumerate(("L", "R")):
            show(fig, glass_tile(tag, K, j, side, d, tuple(col)), xc - (GW - 0.1) / 2, Y_GLASS + (1 - k_) * GLASS_H,
                 GW - 0.1, GLASS_H, W, H)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=5.5, color="#4a5462")
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_xlim(0, PLOT_W)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("% of the cluster's electrodes inside the map", fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{MC.LABEL[tag]}, K = {K}, members with loading ≥ {a.loading:g}: the {M} maps that at least one "
                 "clean cluster fits", fontsize=10.5, loc="left")
    ax.legend(handles=[Patch(facecolor="white", edgecolor=P2.INK, label="fits: beats the shaft-shift null, q < 0.05"),
                       Line2D([0], [0], color="black", lw=1.2, label="all electrodes"),
                       Line2D([0], [0], marker="o", ls="", color=RED, ms=4, label="> 50% one patient")],
              loc="upper right", frameon=False, fontsize=7.5, ncol=3)
    # the selected maps on the pial surface, same order and numbers, left above right
    for k_, m in enumerate(chosen):
        xc = LEFT + (k_ + 0.5) * MAPW
        for s_, side in enumerate(("L", "R")):
            show(fig, pial_tile(MAPS[m][3:], side), xc - MAPW * 0.47, Y_PIAL + (1 - s_) * PIAL_H, MAPW * 0.94, PIAL_H, W, H)
        fig.text(xc / W, (Y_PIAL - 0.05) / H, f"{k_ + 1}  " + NAMES[m].replace(" vs ", "\nvs "), ha="center", va="top",
                 fontsize=6.8, color="#1c2330", linespacing=1.1)
    cax = fig.add_axes([0.25 / W, (Y_PIAL + PIAL_H * 0.6) / H, 0.55 / W, 0.1 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=NS_CMAP), cax=cax, orientation="horizontal",
                      ticks=[0, 1])
    cb.ax.set_xticklabels(["90th\npct", "99.5th\npct+"], fontsize=6)
    cb.ax.tick_params(length=2)
    cax.set_title("coloured =\ntop 10%", fontsize=6.5, pad=2)
    out = OUTD / f"{tag}_K{K}_{LTAG}.png"
    fig.savefig(out, transparent=True)
    plt.close(fig)
    print(f"wrote {out.name} ({W:.0f} x {H:.0f} in): {M} maps shown; fits per cluster: "
          + ", ".join(f"c{j}{'' if clean[j] else '(X)'} {int(fit[j].sum())}" for j in rank)
          + (f"; left out: {', '.join(dropped)}" if dropped else ""))
