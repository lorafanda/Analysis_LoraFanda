"""Step 15 of member_measures (Lora, 2026-11-05): one wide figure per K, the 15-band clusters as rows, every NiMARE
map as a column, every feature set as a bar (matched to the 15-band cluster), at one minimum loading. Reads the cNMF
runs directly (every K in --ks) and step 4's map table; does not depend on --k. Renders with VTK: run with the
desktop unlocked, or --no-render to reuse the saved tiles.

    python 15_cross_feature_maps.py --ns nimare
    python 15_cross_feature_maps.py --ns nimare --ks 7 --loading 0.35 --perm 2000
    python 15_cross_feature_maps.py --ns nimare --no-render          reuse the tiles (no VTK)

    out/cross_feature_maps/K<k>_load035.png     transparent background
    out/cross_feature_maps/K<k>_load035.csv     every bar
    out/cross_feature_maps/tiles/               the glass-brain and pial tiles (reused with --no-render)

LAYOUT, per K. One row per 15-band cluster (the reference), in order of mean P(LanA) of the 15-band members:
    left    the matched cluster of each feature set (15 bands, 15 bands z, 5 bands, 2 bands, HFA) on the glass
            brain, left and right hemisphere, members with own loading >= --loading only, drawn in the row's colour;
            above each pair: the cluster's own id and n (and the match r for the other feature sets)
    right   for each of the 23 maps, one bar per feature set (the pattern tells the feature set, legend), in the
            row's colour:
              bar         % of the matched cluster's members (loading >= --loading) inside the map (the map's top 10%)
              black tick  % of ALL electrodes inside the map (the cohort)
              error bar   95% from bootstrapping patients
              outline     beats the shaft-shift null (memberships rotated along the shaft order within each patient,
                          step 14's null; --perm rotations) at q < 0.05, Benjamini-Hochberg over every bar of the figure
              red dot     more than 50% of those members from one patient
              faded       the match to the 15-band cluster is weak (centred r < --min-r)
    bottom  each map on the pial surface, left hemisphere above right, and its name. A vertex is coloured when the map
            there is in its top 10% (the same 90th-percentile threshold as 'inside' for the electrodes); the colour runs
            from light orange at that threshold to dark red at the map's 99.5th percentile (key under the glass brains).
            Display only: the electrodes use the 5-mm sphere mean at the contact, the surface the vertex value.
MATCHING: FIG 4's rule (00_paper2_figure4_correspondence.py): centroids in the 5-band description space, each
electrode z-scored, members weighted by loading, each solution's mean centroid removed, greedy one-to-one by r."""
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
ap.add_argument("--ks", nargs="+", type=int, default=[6, 7, 8, 9, 10])
ap.add_argument("--loading", type=float, default=0.35)
ap.add_argument("--perm", type=int, default=1000, help="shaft-shift rotations")
ap.add_argument("--min-r", type=float, default=0.5, help="matches below this centred r are drawn faded")
ap.add_argument("--no-render", action="store_true", help="reuse the saved tiles instead of rendering (no VTK)")
a, _ = ap.parse_known_args()

FS = [("concat_bands15", "15band", "15 bands"), ("concat_bands15z", "15zband", "15 bands z"),
      ("concat_bands5", "5band", "5 bands"), ("concat_bands2", "2band", "2 bands"), ("concat_hfa", "hfa", "HFA")]
HATCH = ["", "////", "....", "xxxx", "\\\\\\\\"]
plt.rcParams["hatch.linewidth"] = 0.8
RED = "#d7191c"
LEGEND_FACE = "#5f6670"
ONE_PATIENT = 0.5
LTAG = f"load{int(round(100 * a.loading)):03d}"
P2 = MC.p2()
F4 = MC.load_module(MC.SHARE / "00_paper2_figure4_correspondence.py", "mm_f4")
OUTD = MC.SHARED / "cross_feature_maps"
TILES = OUTD / "tiles"
TILES.mkdir(parents=True, exist_ok=True)
NIM = MC.SHARE.parent / "04_FBM_Pooling" / "neurosynth_nimare"
A305 = np.array([[0.9975, -0.0073, 0.0176, -0.0429], [0.0146, 1.0009, -0.0024, 1.5496],
                 [-0.0130, -0.0093, 0.9971, 1.1840], [0.0, 0.0, 0.0, 1.0]])
NS_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("ns", ["#f7b267", "#d7301f", "#7f0000"])
ns = pd.read_csv(MC.NS_Z).drop(columns="type", errors="ignore").set_index("key")
order = pd.read_csv(MC.NS_MAPS).column.tolist() if MC.NS_MAPS.exists() else []
MAPS = [c for c in order if c in ns.columns] + [c for c in ns.columns if c.startswith("ns_") and c not in order]
NAMES = [m[3:].replace("_vs_", " vs ").replace("_", " ") for m in MAPS]
rng = np.random.default_rng(0)
space = F4.description_space()
Zz = F4.electrode_z(space["Z"])


def shaft_key(name):
    r = re.match(r"^(.*?)(\d+)$", str(name))
    return (r.group(1), int(r.group(2))) if r else (str(name), 0)


def glass_tile(fset, tag, K, j, side, d, colour):
    """Members of cluster j (own loading >= --loading) on the glass brain, in the row's colour."""
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
    """The map on the pial surface where it is in its top 10% (the 'inside' rule of 4_neurosynth.py, at the vertex:
    the maximum over 0, +-1.5 and +-3 mm along the normal, nearest voxel)."""
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


def show(fig, img, x0, y0, w, h, W, H, alpha=1.0):
    """An image placed in inches from the bottom left, aspect kept inside the box."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    ww, hh = iw * s, ih * s
    ax = fig.add_axes([(x0 + (w - ww) / 2) / W, (y0 + (h - hh) / 2) / H, ww / W, hh / H])
    ax.imshow(img, alpha=alpha)
    ax.axis("off")


for K in a.ks:
    # ---- the five solutions at K, their members at the cut, their match to 15 bands
    S, sols = {}, {}
    for fset, tag, label in FS:
        try:
            d = P2.load_run(fset, K)
            s = F4.solution("cnmf", fset, K)
        except SystemExit as e:
            print(f"  skip {tag} K {K}: {e}")
            continue
        F4.align(space, s)
        sols[tag] = dict(C=F4.centroids(Zz, s["W"]))
        keys = [f"{p}|{P2.norm(e)}" for p, e in zip(d["meta"]["patient_id"], d["meta"]["electrode"])]
        Zm = ns.reindex(keys)[MAPS].to_numpy(float)
        has = np.isfinite(Zm).all(1)
        lab = d["lab"]
        load = d["Gn"][np.arange(len(lab)), lab]
        S[tag] = dict(d=d, label=label, fset=fset, lab=lab, has=has, I=(Zm > 0).astype(float),
                      keep=has & (load >= a.loading), pat=d["patient"],
                      sk=[shaft_key(e) for e in d["meta"]["electrode"].to_numpy()])
    if "15band" not in S:
        print(f"  no 15-band run at K {K}, skipped")
        continue
    tags = [t for _, t, _ in FS if t in S]
    match = {"15band": {i: (i, 1.0) for i in range(K)}}
    for tag in tags[1:]:
        R = F4.corr_matrix(sols["15band"]["C"], sols[tag]["C"], centre=True)
        match[tag] = {i: (j, r) for i, j, r in F4.greedy_match(R)}

    # ---- every bar: share, bootstrap interval, one-patient share; then the shaft-shift null over all of them
    ref = S["15band"]
    keys_ref = [f"{p}|{P2.norm(e)}" for p, e in zip(ref["d"]["meta"]["patient_id"], ref["d"]["meta"]["electrode"])]
    P_lana = pd.read_csv(MC.SHARE / "member_measures" / "out_k7" / "15band" / "measures.csv").set_index("key").P_lana \
        .reindex(keys_ref).to_numpy(float)
    rank = sorted(range(K), key=lambda i: -np.nanmean(P_lana[ref["lab"] == i]))
    bars = {}
    pvals = []
    for tag in tags:
        T = S[tag]
        kp = np.where(T["keep"])[0]
        lk, pk, Ik = T["lab"][kp], T["pat"][kp], T["I"][kp]
        pats = np.unique(pk)
        pidx = pd.Series(pk).map({p: i for i, p in enumerate(pats)}).to_numpy()
        groups = [g for g in (np.array(sorted(np.where(pidx == q)[0], key=lambda i: T["sk"][kp[i]]))
                              for q in range(len(pats))) if len(g) > 1]
        share = lambda lb: np.array([Ik[lb == j].mean(0) if (lb == j).any() else np.zeros(len(MAPS)) for j in range(K)])
        obs = share(lk)
        ge = np.zeros_like(obs)
        for _ in range(a.perm):
            perm = np.arange(len(kp))
            for g in groups:
                perm[g] = np.roll(g, rng.integers(1, len(g)))
            ge += share(lk[perm]) >= obs
        p = (ge + 1) / (a.perm + 1)
        coh = T["I"][T["has"]].mean(0)
        B = rng.multinomial(len(pats), np.full(len(pats), 1 / len(pats)), size=1000).astype(float)
        for i in range(K):
            if i not in match[tag]:
                continue
            j, r = match[tag][i]
            sel = lk == j
            n = int(sel.sum())
            if n:
                Sm = np.zeros((len(pats), len(MAPS)))
                np.add.at(Sm, pidx[sel], Ik[sel])
                Nm = np.bincount(pidx[sel], minlength=len(pats)).astype(float)
                den = B @ Nm
                ci = np.quantile((B @ Sm)[den > 0] / den[den > 0, None], [0.025, 0.975], axis=0)
                top = pd.Series(pk[sel]).value_counts().iloc[0] / n
            else:
                ci, top = np.full((2, len(MAPS)), np.nan), 1.0
            bars[(i, tag)] = dict(j=j, r=r, n=n, pct=100 * obs[j], lo=100 * ci[0], hi=100 * ci[1], coh=100 * coh,
                                  one=top > ONE_PATIENT, top=top)
            pvals += [((i, tag, m), p[j, m]) for m in range(len(MAPS))] if n else []
    pv = np.array([x[1] for x in pvals])
    o = np.argsort(pv)
    qv = np.minimum.accumulate((pv[o] * len(pv) / (np.arange(len(pv)) + 1))[::-1])[::-1]
    qq = np.empty_like(qv)
    qq[o] = np.minimum(qv, 1)
    Q = {pvals[k][0]: qq[k] for k in range(len(pvals))}

    # ---- the figure, laid out in inches
    nf, nm = len(tags), len(MAPS)
    BW, BH, GAP = 0.95, 0.95, 0.06                                  # one glass brain, the gap between a pair
    LEFT = nf * (2 * BW + GAP) + (nf - 1) * 0.22 + 0.3
    GW = 1.25                                                       # one map's group of bars
    PLOT_W = nm * GW
    RH, TOP, BOT = 1.55, 0.9, 2.45
    W = LEFT + PLOT_W + 0.8
    H = TOP + K * RH + BOT
    fig = plt.figure(figsize=(W, H), dpi=200)
    fig.patch.set_alpha(0)
    ymax = max([np.nanmax(b["hi"]) for b in bars.values() if b["n"]] + [10]) * 1.05
    w = 0.84 / nf
    rows_csv = []
    for ri, i in enumerate(rank):
        y0 = BOT + (K - 1 - ri) * RH
        col = tuple(np.asarray(P2.cluster_col(i, K))[:3])
        # the glass brains of every feature set's matched cluster
        for fi, tag in enumerate(tags):
            b = bars.get((i, tag))
            if b is None:
                continue
            x0 = 0.15 + fi * (2 * BW + GAP + 0.22)
            weak = b["r"] < a.min_r
            for k_, side in enumerate(("L", "R")):
                img = glass_tile(S[tag]["fset"], tag, K, b["j"], side, S[tag]["d"], col)
                show(fig, img, x0 + k_ * (BW + GAP), y0 + 0.08, BW, BH, W, H, alpha=0.45 if weak else 1.0)
            lab_ = f"c{b['j']}  n {b['n']}" + ("" if tag == "15band" else f"  r {b['r']:.2f}")
            fig.text((x0 + BW + GAP / 2) / W, (y0 + BH + 0.2) / H, lab_, ha="center", va="bottom", fontsize=7,
                     color="#9aa0a6" if weak else (RED if b["one"] else col), fontweight="bold")
        # the bars
        ax = fig.add_axes([LEFT / W, (y0 + 0.12) / H, PLOT_W / W, (RH - 0.3) / H])
        ax.patch.set_alpha(0)
        for mi in range(nm):
            for fi, tag in enumerate(tags):
                b = bars.get((i, tag))
                if b is None or not b["n"]:
                    continue
                x = mi - 0.42 + w * (fi + 0.5)
                weak = b["r"] < a.min_r
                q = Q.get((i, tag, mi), 1.0)
                ax.bar(x, b["pct"][mi], width=w * 0.92, color=col, hatch=HATCH[fi], edgecolor="white", lw=0,
                       alpha=0.4 if weak else 1.0, zorder=2)
                if q < 0.05:
                    ax.bar(x, b["pct"][mi], width=w * 0.92, fill=False, edgecolor=P2.INK, lw=1.1, zorder=3)
                ax.plot([x, x], [b["lo"][mi], b["hi"][mi]], color=P2.INK, lw=0.45, zorder=3)
                if b["one"]:
                    ax.plot(x, min(b["hi"][mi] + 3, ymax * 0.98), "o", color=RED, ms=2.6, zorder=5)
                rows_csv.append(dict(K=K, loading=a.loading, row_15band=i, fset=tag, cluster=b["j"], match_r=round(b["r"], 3),
                                     n=b["n"], top_patient_share=round(b["top"], 2), map=MAPS[mi][3:],
                                     pct_inside=round(b["pct"][mi], 1), lo=round(b["lo"][mi], 1), hi=round(b["hi"][mi], 1),
                                     cohort_pct=round(b["coh"][mi], 1), q_shaft=round(q, 4), fit=bool(q < 0.05)))
            coh = bars[(i, "15band")]["coh"][mi]
            ax.plot([mi - 0.45, mi + 0.45], [coh, coh], color="black", lw=1.0, zorder=4)
        ax.set_xlim(-0.55, nm - 0.45)
        ax.set_ylim(0, ymax)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        for mi in range(1, nm):
            ax.axvline(mi - 0.5, color="#e3e6e4", lw=0.6, zorder=0)
    # the feature-set names over the glass-brain columns
    for fi, tag in enumerate(tags):
        x0 = 0.15 + fi * (2 * BW + GAP + 0.22)
        fig.text((x0 + BW + GAP / 2) / W, (H - TOP + 0.25) / H, S[tag]["label"], ha="center", va="bottom", fontsize=9,
                 fontweight="bold", color="#1c2330")
    # the maps on the pial surface, and their names
    for mi, m in enumerate(MAPS):
        xc = LEFT + (mi + 0.5) * GW
        for k_, side in enumerate(("L", "R")):
            show(fig, pial_tile(m[3:], side), xc - 0.6, 0.62 + (1 - k_) * 0.86, 1.2, 0.84, W, H)   # left above right
        fig.text(xc / W, 0.55 / H, NAMES[mi].replace(" vs ", "\nvs "), ha="center", va="top", fontsize=6.8,
                 color="#1c2330", linespacing=1.1)
    cax = fig.add_axes([1.4 / W, 1.15 / H, 3.4 / W, 0.13 / H])         # the key of the pial colouring
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=NS_CMAP), cax=cax, orientation="horizontal",
                      ticks=[0, 1])
    cb.ax.set_xticklabels(["90th percentile\n(threshold: coloured = inside)", "99.5th percentile\nand above"], fontsize=7)
    cb.ax.tick_params(length=2)
    cax.set_title("functional maps on the pial surface: uncoloured = below the map's top 10%", fontsize=7.5, pad=3)
    fig.text(LEFT / W, (H - 0.3) / H, f"K = {K}, members with loading ≥ {a.loading:g}: % of each matched cluster inside "
             "each functional map", ha="left", va="top", fontsize=11, color="#1c2330")
    fig.text(0.004, (BOT + K * RH / 2) / H, "% inside the map", rotation=90, va="center", fontsize=8,
             transform=fig.transFigure)
    fig.legend(handles=[Patch(facecolor=LEGEND_FACE, hatch=HATCH[fi] * 2, edgecolor="white", label=S[t]["label"])
                        for fi, t in enumerate(tags)]
               + [Line2D([0], [0], marker="o", ls="", color=RED, ms=4, label="> 50% one patient"),
                  Patch(facecolor="white", edgecolor=P2.INK, label="beats the shaft-shift null, q < 0.05"),
                  Line2D([0], [0], color="black", lw=1.2, label="all electrodes")],
               loc="upper right", bbox_to_anchor=(1 - 0.3 / W, 1 - 0.12 / H), ncol=nf + 3, frameon=False, fontsize=8,
               handlelength=2.4, handleheight=1.3)
    out = OUTD / f"K{K}_{LTAG}.png"
    fig.savefig(out, transparent=True)
    plt.close(fig)
    pd.DataFrame(rows_csv).to_csv(OUTD / f"K{K}_{LTAG}.csv", index=False)
    nfit = sum(1 for r in rows_csv if r["fit"])
    print(f"wrote {out.name} ({W:.0f} x {H:.0f} in): {nfit} of {len(rows_csv)} bars fit")
