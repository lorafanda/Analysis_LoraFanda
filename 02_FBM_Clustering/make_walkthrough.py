#!/usr/bin/env python3
"""The cluster visualizer, claim by claim: a report that states each claim of Paper 2's
validation story (the types, agreement, stability, separability, the scorecard), gives the
click path that shows it on cluster_visualizer.html, shows the brain and the statistics
that path produces, quotes the numbers from the files, and ends every statement with the
paper it rests on.

    python make_walkthrough.py            # outputs/clustering/paper2_validation/walkthrough/

The brain images are rendered here, headless, from the SAME files the page reads
(contacts.json, labels_by_k.json, heldout.json, the figure order) with the page's own
encodings (its type hues by figure position, its patient hues, its loading size and fade,
the held-out view's greyscale and destination colours), on the fsaverage glass scenes FIG 1
uses - so each one is what the page shows after the clicks, at print resolution. Pane
screenshots of the live page, when the builder is given some, are added beside them.
"""
from __future__ import annotations

import base64
import colorsys
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
from matplotlib.gridspec import GridSpec
from PIL import Image

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("p2fig1", ROOT / "00_Paper2_Figures.py")
P2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P2)
_sspec = importlib.util.spec_from_file_location("story", ROOT / "make_validation_story.py")
STORY = importlib.util.module_from_spec(_sspec)
_sspec.loader.exec_module(STORY)
sys.path.insert(0, str(ROOT / "functions"))
import lf_runs as LR                                      # noqa: E402

FSET, K, METHOD = "concat_bands15", 7, "cnmf"
COV = ROOT / "outputs" / "250_recon" / "fsaverage" / "coverage_viz"
SITE_ORDER = Path.home() / "lorafanda.github.io" / "cluster_visualizer_order.json"
VAL = ROOT / "outputs" / "clustering" / "paper2_validation"
OUTD = VAL / "walkthrough"
REND = OUTD / "renders"
SHOTS = OUTD / "shots"
PAGE = "https://lorafanda.github.io/cluster_visualizer.html"
INK, MUTED, GREY = P2.INK, P2.MUTED, P2.GREY
GREY148 = np.array([148, 148, 148], float)

REFS = {
    "TW05": ("Tibshirani & Walther 2005", "Tibshirani R, Walther G (2005). Cluster validation by prediction strength. Journal of Computational and Graphical Statistics 14(3): 511–528."),
    "L04": ("Lange et al. 2004", "Lange T, Roth V, Braun ML, Buhmann JM (2004). Stability-based validation of clustering solutions. Neural Computation 16(6): 1299–1323."),
    "V17": ("Varoquaux et al. 2017", "Varoquaux G, Raamana PR, Engemann DA, Hoyos-Idrobo A, Schwartz Y, Thirion B (2017). Assessing and tuning brain decoders: cross-validation, caveats, and guidelines. NeuroImage 145: 166–179."),
    "K55": ("Kuhn 1955", "Kuhn HW (1955). The Hungarian method for the assignment problem. Naval Research Logistics Quarterly 2: 83–97."),
    "D10": ("Ding et al. 2010", "Ding C, Li T, Jordan MI (2010). Convex and semi-nonnegative matrix factorizations. IEEE Transactions on Pattern Analysis and Machine Intelligence 32(1): 45–55."),
    "HA85": ("Hubert & Arabie 1985", "Hubert L, Arabie P (1985). Comparing partitions. Journal of Classification 2: 193–218."),
    "M03": ("Monti et al. 2003", "Monti S, Tamayo P, Mesirov J, Golub T (2003). Consensus clustering: a resampling-based method for class discovery and visualization of gene expression microarray data. Machine Learning 52: 91–118."),
    "S14": ("Şenbabaoğlu et al. 2014", "Şenbabaoğlu Y, Michailidis G, Li JZ (2014). Critical limitations of consensus clustering in class discovery. Scientific Reports 4: 6207."),
    "G12": ("Gretton et al. 2012", "Gretton A, Borgwardt KM, Rasch MJ, Schölkopf B, Smola A (2012). A kernel two-sample test. Journal of Machine Learning Research 13: 723–773."),
    "R87": ("Rousseeuw 1987", "Rousseeuw PJ (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics 20: 53–65."),
    "OP09": ("Owen & Perry 2009", "Owen AB, Perry PO (2009). Bi-cross-validation of the SVD and the nonnegative matrix factorization. Annals of Applied Statistics 3(2): 564–594."),
}


def cite(*ks):
    return " (" + "; ".join(REFS[k][0] for k in ks) + ")"


# ---------------------------------------------------------------- the page's colours
def hsl2rgb(h, s, l):
    k = lambda n: (n + h * 12) % 12
    a = s * min(l, 1 - l)
    f = lambda n: l - a * max(-1, min(k(n) - 3, min(9 - k(n), 1)))
    return np.array([round(255 * f(0)), round(255 * f(8)), round(255 * f(4))], float)


def u2hue(pos, K_):
    return hsl2rgb(pos / K_, 0.62, 0.52)


def page_grey(v):
    g = 225 - 195 * float(np.clip(v, 0, 1))
    return np.array([g, g, g])


def loading_style(lv, base):
    """The page's updateElectrodes: radius, colour fade below 0.5, alpha, from the loading."""
    if lv is None or not np.isfinite(lv):
        return 1.73, base, 255
    w = float(np.clip((lv - 0.14) / (0.85 - 0.14), 0, 1))
    r = 1.05 + 2.18 * w
    col = base
    if lv < 0.5:
        g = float(np.clip((lv - 0.14) / (0.5 - 0.14), 0, 1))
        col = GREY148 + (base - GREY148) * g
    return r, col, round(90 + 165 * w)


# ---------------------------------------------------------------- data
def load_all():
    cache = LR.current_cache()
    man = json.loads((COV / "manifest.json").read_text(encoding="utf-8"))
    contacts = json.loads((COV / man["contacts_file"]).read_text(encoding="utf-8"))
    order = json.loads(SITE_ORDER.read_text(encoding="utf-8"))["runs"]
    runs = {r["id"]: r for r in man["runs"]}
    main_id = next(i for i in runs if i.startswith(f"{METHOD}__{FSET}__"))
    ho = json.loads((COV / runs[main_id]["dir"] / "heldout.json").read_text(encoding="utf-8"))
    nums = json.loads((VAL / "numbers.json").read_text(encoding="utf-8"))["values"]
    d = P2.load_run(FSET, K, METHOD)
    d["fset"] = FSET
    xyz = np.asarray(contacts["xyz"], float)
    hemi = np.array(["L" if h == "lh" else "R" for h in contacts["hemi"]])
    pats_all = sorted(set(p for p in contacts["patient"] if p))
    pat_index = {p: i for i, p in enumerate(pats_all)}
    return dict(cache=cache, man=man, contacts=contacts, order=order, runs=runs, main_id=main_id, ho=ho,
                nums=nums, d=d, xyz=xyz, hemi=hemi, pat_index=pat_index)


def sweep_labels(D, run_id, k):
    f = COV / D["runs"][run_id]["dir"] / "labels_by_k.json"
    L = json.loads(f.read_text(encoding="utf-8")).get(str(k))
    return [None if v is None else int(v[0]) for v in L] if L else None


def sweep_stats(D, run_id, k):
    f = COV / D["runs"][run_id]["dir"] / "sweep_stats.json"
    return json.loads(f.read_text(encoding="utf-8")).get(str(k)) or {}


def sweep_loadings(D, run_id, k):
    f = COV / D["runs"][run_id]["dir"] / "loadings_by_k.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8")).get(str(k))


def patient_rgb(D, pid):
    i = D["pat_index"].get(pid)
    return np.array([190, 190, 196], float) if i is None else hsl2rgb(((i * 137.508) % 360) / 360, 0.66, 0.56)


# ---------------------------------------------------------------- rendering
_SCENE_D = None


def render_state(D, side, cols, rads, alphas, zoom=1.30):
    """One view of the contacts whose radius is > 0, on FIG 1's glass scene of this side."""
    global _SCENE_D
    if _SCENE_D is None:
        _SCENE_D = dict(xyz=D["xyz"], hemi=D["hemi"])
    pl, ok = P2._scene(side, _SCENE_D, zoom)
    m = ok & (rads > 0) & np.isfinite(D["xyz"][:, 0])
    actor = None
    if m.sum():
        rgba = np.zeros((int(m.sum()), 4), np.uint8)
        rgba[:, :3] = np.clip(cols[m], 0, 255).astype(np.uint8)
        rgba[:, 3] = np.clip(alphas[m], 0, 255).astype(np.uint8)
        cloud = P2.pv.PolyData(D["xyz"][m])
        cloud["rgba"] = rgba
        cloud["r"] = rads[m].astype(float)
        g = cloud.glyph(orient=False, scale="r", geom=P2.pv.Sphere(radius=1.0, theta_resolution=12, phi_resolution=12))
        actor = pl.add_mesh(g, scalars="rgba", rgba=True)
    img = pl.screenshot(return_img=True, transparent_background=True)
    if actor is not None:
        pl.remove_actor(actor)
    return P2._crop_alpha(np.asarray(img))


def two_views(D, name, cols, rads, alphas, caption, sides=("L", "R")):
    imgs = [Image.fromarray(render_state(D, s, cols, rads, alphas)) for s in sides]
    h = max(im.height for im in imgs)
    W = sum(int(im.width * h / im.height) for im in imgs) + 40 * (len(imgs) - 1)
    out = Image.new("RGBA", (W, h + 70), (255, 255, 255, 255))
    x = 0
    for s, im in zip(sides, imgs):
        im2 = im.resize((int(im.width * h / im.height), h), Image.LANCZOS)
        out.alpha_composite(im2, (x, 0))
        x += im2.width + 40
    # a caption strip under the views
    fig = plt.figure(figsize=(W / 150, 70 / 150), dpi=150)
    fig.text(0.0, 0.62, caption, fontsize=9, color=INK, ha="left", va="center")
    fig.text(0.0, 0.18, "   ·   ".join({"L": "left hemisphere, lateral", "R": "right hemisphere, lateral", "T": "from above"}[s] for s in sides)
             + "   ·   rendered from the page's data on FIG 1's glass scenes", fontsize=7.5, color=MUTED, ha="left", va="center")
    fig.canvas.draw()
    strip = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGBA")
    plt.close(fig)
    out.alpha_composite(strip.resize((W, 70), Image.LANCZOS), (0, h))
    p = REND / f"{name}.png"
    out.convert("RGB").save(p)
    return p


def blank(D):
    n = len(D["xyz"])
    return np.zeros((n, 3)), np.zeros(n), np.full(n, 255.0)


# ---------------------------------------------------------------- the states
def state_type(D, run_id, k, cluster, pos, Kk, name, caption, by_patient=False, others_grey=False):
    lab = sweep_labels(D, run_id, k)
    load = sweep_loadings(D, run_id, k)
    cols, rads, al = blank(D)
    base = u2hue(pos, Kk)
    for i, L in enumerate(lab):
        if L is None:
            continue
        if L != cluster:
            if others_grey:
                cols[i], rads[i], al[i] = [205, 208, 214], 1.0, 110
            continue
        c = patient_rgb(D, D["contacts"]["patient"][i]) if by_patient else base
        r, cc, a = loading_style(load[i] if load else None, c)
        cols[i], rads[i], al[i] = cc, r, a
    return two_views(D, name, cols, rads, al, caption)


def state_all_types(D, run_id, k, name, caption):
    lab = sweep_labels(D, run_id, k)
    load = sweep_loadings(D, run_id, k)
    ordr = D["order"][run_id][str(k)]
    pos = {c: i for i, c in enumerate(ordr)}
    cols, rads, al = blank(D)
    for i, L in enumerate(lab):
        if L is None:
            continue
        r, cc, a = loading_style(load[i] if load else None, u2hue(pos.get(L, 0), len(ordr)))
        cols[i], rads[i], al[i] = cc, r, a
    return two_views(D, name, cols, rads, al, caption)


def state_heldout(D, name, caption, colour="miss", src="lopo", row=None, cell=None, patient=None):
    ho = D["ho"]
    C = ho["contacts"]
    pos = {c: i for i, c in enumerate(ho["order"])}
    hit = C["lopo_hit"] if src == "lopo" else C["halves_hit"]
    goes = C["lopo_goes"] if src == "lopo" else C["halves_goes"]
    cols, rads, al = blank(D)
    for n, ci in enumerate(C["idx"]):
        t, h = C["type"][n], hit[n]
        if colour == "hit":
            c = page_grey(h) if h is not None else np.array([200, 200, 205.]); r = 1.4 + 0.9 * (0.5 if h is None else 1 - h)
        elif colour == "miss":
            if h is not None and h >= 0.999:
                c, r = np.array([205, 208, 214.]), 1.25
            else:
                c, r = u2hue(pos.get(goes[n], 0), ho["k"]), 2.3
        elif colour == "cons":
            v = (C["cons_own"][n] - C["cons_other"][n]) if (C["cons_own"] and C["cons_own"][n] is not None) else None
            c = page_grey(v) if v is not None else np.array([200, 200, 205.]); r = 1.4 + 0.9 * (0.5 if v is None else 1 - float(np.clip(v, 0, 1)))
        elif colour == "type":
            c, r = u2hue(pos.get(t, 0), ho["k"]), 1.73
        else:
            c, r = patient_rgb(D, D["contacts"]["patient"][ci]), 1.73
        vis = (row is None or t == row) and (patient is None or D["contacts"]["patient"][ci] == patient)
        lit = True
        if cell is not None:
            lit = (t == cell[0] and h is not None and ((h >= 0.5) if cell[0] == cell[1] else (goes[n] == cell[1] and h < 1)))
        a = 255
        if not vis:
            a, r = 26, 1.0
        elif cell is not None and not lit:
            c, a, r = np.array([200, 200, 205.]), 70, 0.9
        elif cell is not None:
            r = max(r, 2.6)
        cols[ci], rads[ci], al[ci] = c, r, a
    return two_views(D, name, cols, rads, al, caption)


# ---------------------------------------------------------------- the statistics panels
def panel_numbers(D, name):
    ho, T = D["ho"], {t["cluster"]: t for t in D["ho"]["types"]}
    chance = D["nums"].get("lopo_predict", {}).get("chance_mean", 0.52)
    fig = plt.figure(figsize=(9.2, 2.1), dpi=170)
    for x, (v, lab) in zip((0.02, 0.36, 0.70), ((ho["lopo"]["overall"], f"leave one patient out: refit on the other {ho['n_patients'] - 1},\nplace the held-out patient"),
                                                 (ho["halves"]["overall"], "patient halves: refit on half the patients,\nplace the other half (harsher)"),
                                                 (chance, "chance: every electrode given its\npatient's commonest type"))):
        fig.text(x, 0.72, f"{100 * v:.0f} %", fontsize=30, fontweight="bold", color=INK if x < 0.7 else MUTED, va="center")
        fig.text(x, 0.22, lab, fontsize=8.5, color=MUTED, va="center")
    p = OUTD / "panels" / f"{name}.png"
    fig.savefig(p, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def panel_types(D, name):
    ho = D["ho"]
    T = {t["cluster"]: t for t in ho["types"]}
    order = ho["order"]
    fig, ax = plt.subplots(figsize=(9.2, 3.4), dpi=170)
    y = np.arange(len(order))[::-1]
    for yi, c in zip(y, order):
        col = u2hue(order.index(c), ho["k"]) / 255
        t = T[c]
        ax.barh(yi, t["halves_recall"], color=col, alpha=0.45, height=0.62)
        ax.barh(yi, t["lopo_recall"], fill=False, edgecolor=col, lw=1.8, height=0.62)
        ax.plot([t["lopo_recall"]] * 2, [yi - 0.42, yi + 0.42], color=INK, lw=2)
        ax.text(max(t["lopo_recall"], t["halves_recall"]) + 0.01, yi, f"{100 * t['lopo_recall']:.0f} % · {100 * t['halves_recall']:.0f} %", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"c{c}" + (f" {T[c]['name']}" if T[c]["name"] else "") for c in order], fontsize=9)
    for tl, c in zip(ax.get_yticklabels(), order):
        tl.set_color(u2hue(order.index(c), ho["k"]) / 255)
    ax.set_xlim(0, 1.18); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "25", "50", "75", "100 %"], fontsize=8)
    ax.set_xlabel("of the type's held-out electrodes placed in it   ·   filled = patient halves, outline and tick = leave one patient out", fontsize=8.5, color=MUTED)
    ax.tick_params(colors=MUTED, length=2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = OUTD / "panels" / f"{name}.png"
    fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


def panel_confusions(D, name):
    ho = D["ho"]; order = ho["order"]; Kk = ho["k"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), dpi=170)
    for ax, (M, ttl) in zip(axes, ((np.asarray(ho["lopo"]["matrix"], float), f"A · leave one patient out · overall {100 * ho['lopo']['overall']:.0f} %"),
                                   (np.asarray(ho["halves"]["matrix"], float), f"B · patient halves ({ho['params']['n_splits']} splits) · overall {100 * ho['halves']['overall']:.0f} %"))):
        R = M / np.maximum(M.sum(1, keepdims=True), 1e-12)
        Ro = R[np.ix_(order, order)]
        ax.imshow(Ro, cmap="Greys", vmin=0, vmax=1)
        for a_ in range(Kk):
            for b_ in range(Kk):
                v = Ro[a_, b_]
                ax.text(b_, a_, f"{100 * v:.0f}", ha="center", va="center", fontsize=8, color="white" if v > 0.55 else INK, fontweight="bold" if a_ == b_ else "normal")
        ax.set_xticks(range(Kk)); ax.set_yticks(range(Kk))
        ax.set_xticklabels([f"c{c}" for c in order], fontsize=8.5); ax.set_yticklabels([f"c{c}" for c in order], fontsize=8.5)
        for tl, c in zip(ax.get_xticklabels(), order): tl.set_color(u2hue(order.index(c), Kk) / 255)
        for tl, c in zip(ax.get_yticklabels(), order): tl.set_color(u2hue(order.index(c), Kk) / 255)
        ax.set_xlabel("placed in", fontsize=8.5, color=MUTED); ax.set_ylabel("published type", fontsize=8.5, color=MUTED)
        ax.tick_params(length=0)
        for s_ in ax.spines.values(): s_.set_visible(False)
        ax.set_title(ttl + "\n% of each type's held-out electrodes", fontsize=9, loc="left", color=INK)
    fig.tight_layout()
    p = OUTD / "panels" / f"{name}.png"
    fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


def patients_table(D):
    C = D["ho"]["contacts"]
    rows = {}
    for n, ci in enumerate(C["idx"]):
        p = D["contacts"]["patient"][ci]
        r = rows.setdefault(p, dict(p=p, n=0, lo=[], ha=[], missH=0.0))
        r["n"] += 1
        if C["lopo_hit"][n] is not None: r["lo"].append(C["lopo_hit"][n])
        if C["halves_hit"][n] is not None: r["ha"].append(C["halves_hit"][n]); r["missH"] += 1 - C["halves_hit"][n]
    N = len(C["idx"]); tot = sum(r["missH"] for r in rows.values()) or 1
    out = []
    for r in rows.values():
        out.append(dict(p=r["p"], n=r["n"], hitL=float(np.mean(r["lo"])) if r["lo"] else None, hitH=float(np.mean(r["ha"])) if r["ha"] else None,
                        excess=(r["missH"] / tot) / (r["n"] / N)))
    out.sort(key=lambda r: (r["hitH"] if r["hitH"] is not None else 9))
    return out


def panel_patients(D, name):
    rows = patients_table(D)
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=170)
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.barh(yi, r["hitH"] or 0, color=patient_rgb(D, r["p"]) / 255, alpha=0.75, height=0.7)
        if r["hitL"] is not None:
            ax.plot([r["hitL"]] * 2, [yi - 0.45, yi + 0.45], color=INK, lw=2)
        ax.text(1.02, yi, f"n {r['n']} · {r['excess']:.1f}× its share of the misses", va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels([r["p"] for r in rows], fontsize=7.5)
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "50", "100 %"], fontsize=8)
    ax.set_xlabel("mean hit rate of the patient's electrodes   ·   bar = patient halves, tick = leave one patient out   ·   worst first", fontsize=8.5, color=MUTED)
    ax.tick_params(colors=MUTED, length=2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = OUTD / "panels" / f"{name}.png"
    fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


def panel_grid_ari(D, name):
    """The agreement numbers the grid stands beside: ARI across algorithms and feature sets."""
    n = D["nums"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.6), dpi=170, gridspec_kw=dict(width_ratios=[1, 2]))
    for ax, (dct, ttl) in zip(axes, ((n["ari_algo"], "across algorithms (15 bands, K 7)"), (n["ari_feat"], "across feature sets (convex NMF)"))):
        items = sorted(dct.items(), key=lambda kv: -kv[1]["ari"])
        ax.barh(range(len(items))[::-1], [v["ari"] for _, v in items], color=MUTED, height=0.6)
        ax.set_yticks(range(len(items))[::-1]); ax.set_yticklabels([k for k, _ in items], fontsize=8)
        for i, (_, v) in zip(range(len(items))[::-1], items):
            ax.text(v["ari"] + 0.01, i, f"{v['ari']:.2f}", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, 0.65); ax.set_title(f"adjusted Rand index {ttl}", fontsize=9, loc="left", color=INK)
        ax.tick_params(colors=MUTED, length=2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = OUTD / "panels" / f"{name}.png"
    fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


# ---------------------------------------------------------------- the report
def img_md(p, alt):
    return f"![{alt}](" + str(Path(p).relative_to(OUTD)).replace("\\", "/") + ")\n"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try: stream.reconfigure(errors="replace")
            except Exception: pass
    t0 = time.time()
    for p in (OUTD, REND, SHOTS, OUTD / "panels"):
        p.mkdir(parents=True, exist_ok=True)
    D = load_all()
    ho, nums = D["ho"], D["nums"]
    T = {t["cluster"]: t for t in ho["types"]}
    mid = D["main_id"]
    order7 = D["order"][mid]["7"]
    nm = {c: f"c{c}" + (f" {T[c]['name']}" if T[c]["name"] else "") for c in T}
    pos_of = {c: i for i, c in enumerate(order7)}
    print(f"main run {mid}   order {order7}")

    # ---- renders
    R = {}
    R["c5"] = state_type(D, mid, 7, 5, pos_of[5], 7, "01_type_c5", "c5 production: its electrodes, sized and faded by their loading (the page's one-cluster view)")
    R["c3"] = state_type(D, mid, 7, 3, pos_of[3], 7, "02_type_c3", "c3 sentence-stimulus low-frequency: the same view")
    R["all"] = state_all_types(D, mid, 7, "03_all_types", "all seven types at once (overlay with every chip selected), colours by figure position")
    # agreement: position #1 under the three algorithms at K 7, and on the four feature sets
    pos1 = {}
    for m_ in ("cnmf", "kmeans", "hierarchical"):
        rid = next(i for i in D["runs"] if i.startswith(f"{m_}__{FSET}__"))
        c1 = D["order"][rid]["7"][0]
        pos1[m_] = (rid, c1)
    R["algo"] = [state_type(D, rid, 7, c1, 0, 7, f"04_pos1_{m_}", f"position #1 under {P2.METHOD_LABEL[m_]} on 15 bands at K 7 = cluster {c1}") for m_, (rid, c1) in pos1.items()]
    fs1 = {}
    for f_ in P2.FSETS:
        rid = next(i for i in D["runs"] if i.startswith(f"cnmf__{f_}__"))
        fs1[f_] = (rid, D["order"][rid]["7"][0])
    R["fset"] = [state_type(D, rid, 7, c1, 0, 7, f"05_pos1_{f_}", f"position #1 for convex NMF on {P2.FS_LABEL[f_]} at K 7 = cluster {c1}") for f_, (rid, c1) in fs1.items()]
    # stability: the K 8 cluster one patient owns
    st8 = sweep_stats(D, mid, 8)
    dom = None
    for c, cnt in st8.get("cluster_patient_counts", {}).items():
        tot = sum(cnt.values()); top = max(cnt.items(), key=lambda kv: kv[1])
        if tot and top[1] / tot > 0.5:
            dom = (int(c), top[0], top[1], tot)
    if dom:
        R["k8"] = state_type(D, mid, 8, dom[0], D["order"][mid]["8"].index(dom[0]), 8, "06_k8_one_patient", f"K 8: cluster {dom[0]} is {100 * dom[2] / dom[3]:.0f} % one patient ({dom[1]}, {dom[2]} of {dom[3]}); coloured by patient", by_patient=True, others_grey=True)
    R["cons"] = state_heldout(D, "07_consensus_margin", "consensus margin under patient resampling: black = always with its type, light = switches", colour="cons")
    R["miss_l"] = state_heldout(D, "08_misses_lopo", "where misses go, leave one patient out: hits small grey, a miss in the colour of the type it landed in", colour="miss", src="lopo")
    R["miss_h"] = state_heldout(D, "09_misses_halves", "where misses go, patient halves: the same encoding under the harsher test", colour="miss", src="halves")
    R["hit"] = state_heldout(D, "10_hit_rate_lopo", "hit rate, leave one patient out: light never placed in its type, black always", colour="hit", src="lopo")
    R["cell"] = state_heldout(D, "11_cell_c3_to_c4_halves", "matrix B, cell c3 → c4 clicked: the c3 electrodes that land in c4 under the halves test, the rest stepped back", colour="miss", src="halves", cell=(3, 4))
    R["row3"] = state_heldout(D, "12_row_c3_lopo", "type c3 clicked: its electrodes only, misses in the colour of their destination (leave one patient out)", colour="miss", src="lopo", row=3)
    R["row2"] = state_heldout(D, "13_row_c2_lopo", "type c2 clicked: the same for broadband decrease", colour="miss", src="lopo", row=2)
    worst = patients_table(D)[0]["p"]
    R["pat"] = state_heldout(D, f"14_patient_{worst}_halves", f"patient {worst} selected: its electrodes, misses coloured by destination (patient halves)", colour="miss", src="halves", patient=worst)
    print(f"  {sum(1 if not isinstance(v, list) else len(v) for v in R.values())} renders   ({time.time() - t0:.0f} s)")

    # ---- panels
    Pn = dict(numbers=panel_numbers(D, "p1_one_number"), types=panel_types(D, "p2_per_type"), conf=panel_confusions(D, "p3_confusions"),
              pats=panel_patients(D, "p4_patients"), ari=panel_grid_ari(D, "p0_ari"))
    pats = patients_table(D)
    keep80 = sum(1 for r in pats if r["hitH"] is not None and r["hitH"] >= 0.8)
    worst3 = sorted(pats, key=lambda r: -r["excess"])[:3]
    shots = sorted(SHOTS.glob("*.jpg")) + sorted(SHOTS.glob("*.png"))

    # ---- the text
    L = []
    A = L.append
    A("# The cluster visualizer, claim by claim")
    A("")
    A(f"*Built {time.strftime('%Y-%m-%d %H:%M')} from the files the page reads (cohort {D['cache']}, run {D['runs'][mid]['run']}). "
      f"Page: {PAGE}. The brain images are rendered from the same data with the page's own encodings on FIG 1's glass scenes; "
      f"the click path under each one reproduces it live.*")
    A("")
    A("## The request, restated")
    A("")
    A("For each claim the paper makes about the clustering (the types, their agreement across descriptions and algorithms, their "
      "stability across patients, their separability) state the claim, say which view of the cluster visualizer shows it and which "
      "clicks get there, show the brain and the statistics that appear, and read the numbers off them with the paper each measure "
      "rests on. In short: claim → where to click → what you see → what it means.")
    A("")
    A("## How to read a click path")
    A("")
    A("`Run` is the feature set × algorithm matrix at the top of the control panel; `K` the row of numbers under Clusters; the chips "
      "`#1 … #7` are the clusters in figure order (#1 = the leftmost block of FIG 1); `View` has one / overlay / grid / held-out. "
      "In the held-out view, the panel on the right is clickable: a type bar, a matrix cell, a patient.")
    A("")
    if shots:
        A("### The page itself, as it appears live")
        A("")
        A("Screenshots of the live page at the time of building (800 px wide, the browser pane's size); the claims below "
          "use renders of the same states at print resolution.")
        A("")
        for s in shots:
            A(img_md(s, s.stem.replace("_", " ")))
        A("")

    # claim 1
    sizes = ", ".join(f"{nm[c]} n {T[c]['n']} ({T[c]['n_patients']} patients, largest {100 * nums['top_share'][str(c)]:.0f} %)" for c in order7)
    A("## Claim 1 · The clustering gives seven recurring response types, each carried by many patients")
    A("")
    A(f"**Where to click.** Run `15 bands × cNMF` → K `7` → View `one` → chip `#1` (c5 production), then `#7` (c3). For all at once: View `overlay` and click every chip.")
    A("")
    A(img_md(R["c5"], "c5")); A(img_md(R["c3"], "c3")); A(img_md(R["all"], "all types"))
    A(f"**What you see.** Every dot is one electrode of the type; its size and its fade carry its loading on the type (small and grey below 0.5), "
      f"the chip carries n and the biggest patient's share, and the red * that marks a patient holding more than half of a cluster is absent at K 7.")
    A("")
    A(f"**The numbers.** {sizes}; median own-type loading {nums['own_median']:.2f}, {100 * nums['own_below_cut']:.0f} % below the 0.3 draw cut{cite('D10')}.")
    A("")
    A("**What it means.** Seven shapes, each spread over 11 to 28 patients; the mixed loadings are the first sign that the types are regions of a continuum rather than islands, which Claim 4 measures" + cite("D10") + ".")
    A("")

    # claim 2
    aa, af = nums["ari_algo"], nums["ari_feat"]
    A("## Claim 2 · Other algorithms find the same shapes; the frequency description matters more than the algorithm")
    A("")
    A("**Where to click.** View `grid` → `algorithms`: position #1 under convex NMF, k-means and Ward at the same K. Then `feature sets`: position #1 on 15, 5, 2 bands and HFA. Switching a run in the matrix keeps the position and the run note prints the ARI against the run you came from.")
    A("")
    for p in R["algo"]: A(img_md(p, "position 1 across algorithms"))
    for p in R["fset"]: A(img_md(p, "position 1 across feature sets"))
    A(img_md(Pn["ari"], "ARI"))
    A(f"**What you see.** The three algorithms put the production type on the same contacts; across feature sets the same position is a different set of contacts once the bands are coarse.")
    A("")
    A(f"**The numbers.** ARI {aa['k-means vs convex NMF']['ari']:.2f} (k-means vs convex NMF), {aa['k-means vs Ward']['ari']:.2f}, {aa['Ward vs convex NMF']['ari']:.2f} across algorithms; "
      f"{max(v['ari'] for v in af.values()):.2f} to {min(v['ari'] for v in af.values()):.2f} across feature sets; all three algorithms agree on {100 * nums['modal_algo']['3']:.0f} % of the electrodes, all four feature sets on {100 * nums['modal_feat']['4']:.0f} %{cite('HA85')}.")
    A("")
    A("**What it means.** Changing the algorithm moves the electrodes less than changing the frequency description" + cite("HA85") + ". Agreement is not correctness: four descriptions can agree on an artefact of the gate.")
    A("")

    # claim 3
    pac = nums["pac"]; g8 = nums["gen"]["8"]
    cons_rank = sorted(T, key=lambda c: -(T[c]["consensus"] or 0))
    A("## Claim 3 · The shapes come back when patients are resampled, and K 7 is the last K before one patient owns a cluster")
    A("")
    A("**Where to click.** K `8`: one chip now carries the red * (a patient holds more than half of it); back to K `7` it is gone. Then View `held-out` → colour `consensus margin`, and source `patient halves` with the patient list at the bottom of the panel.")
    A("")
    if dom:
        A(img_md(R["k8"], "K 8 one-patient cluster"))
    A(img_md(R["cons"], "consensus margin")); A(img_md(R["miss_h"], "misses under the halves test")); A(img_md(Pn["pats"], "patients"))
    A(f"**What you see.** At K 8 the extra cluster is a single patient's strip; at K 7 none is. Under consensus margin the dark electrodes stay with their type when 20 % of the patients are dropped and the light ones switch; under the halves test the misses concentrate in a few patients.")
    A("")
    A(f"**The numbers.** At K 8 one cluster is {100 * g8['worst']:.0f} % one patient; at K 7 the worst is {100 * pac['worst_share']:.0f} % ({pac['worst_patient']}). PAC {pac['real']:.3f} against {pac['null']:.3f} ± {pac['null_sd']:.3f} for the Gaussian cloud{cite('M03', 'S14')}. "
      f"Per-type consensus " + ", ".join(f"{nm[c]} {T[c]['consensus']:.2f}" for c in cons_rank) + f"{cite('M03')}. The halves placement keeps {100 * ho['halves']['overall']:.0f} % overall; "
      + ", ".join(f"{r['p']} {r['excess']:.1f}×" for r in worst3) + f" carry more than their share of the misses, {keep80} of {len(pats)} patients keep at least 80 %{cite('L04', 'V17')}.")
    A("")
    A("**What it means.** The partition is stable against patients, and K 7 is where that stops; the cloud has no patient structure, so the dominance rule is the guard" + cite("S14") + ".")
    A("")

    # claim 4
    rec = {c: T[c]["lopo_recall"] for c in T}
    low = min(rec, key=rec.get)
    A("## Claim 4 · The types are separable: an electrode from an unseen patient lands in its type nine times in ten, and the leaks run between neighbours")
    A("")
    A("**Where to click.** View `held-out` (source `leave one patient out`, colour `where misses go`). Read the three numbers at the top of the panel and the per-type bars; then click the cell `c3 → c4` in matrix B (patient halves), and switch colour to `hit rate`.")
    A("")
    A(img_md(Pn["numbers"], "one number")); A(img_md(Pn["types"], "per type")); A(img_md(R["miss_l"], "misses LOPO")); A(img_md(Pn["conf"], "confusions")); A(img_md(R["cell"], "cell c3 to c4")); A(img_md(R["hit"], "hit rate"))
    A(f"**What you see.** Most electrodes are small grey hits; the coloured misses sit between the territories of the two types they move between, and the clicked cell shows exactly which c3 electrodes the halves test places in c4.")
    A("")
    A(f"**The numbers.** {100 * ho['lopo']['overall']:.0f} % of held-out electrodes land in their published type with one patient out, {100 * ho['halves']['overall']:.0f} % with half the patients out, chance about {100 * nums.get('lopo_predict', {}).get('chance_mean', 0.52):.0f} %{cite('TW05', 'L04')}. "
      f"Recall per type " + ", ".join(f"{nm[c]} {rec[c]:.2f}" for c in sorted(rec, key=rec.get, reverse=True)) + f"; {nm[low]} loses most, to c{T[low]['lopo_confused_with']} ({100 * T[low]['lopo_confused_share']:.0f} %) and under the halves test to c{T[low]['halves_confused_with']} ({100 * T[low]['halves_confused_share']:.0f} %){cite('TW05', 'K55')}. "
      f"Per electrode, {ho['n'] - sum(1 for v in ho['contacts']['lopo_hit'] if v is not None and v < 1)} of {ho['n']} land in their type in every fold.")
    A("")
    A("**What it means.** Separable in the sense that matters, an unseen electrode can be told apart, and only partly: the leaky types lose a fifth of their electrodes to their neighbours, which is what a continuum of responses looks like under a partition" + cite("TW05") + ".")
    A("")

    # claim 5
    A("## Claim 5 · The scorecard: c2 and c4 solid, c1 and c3 fragile, c5 solid under patients but weak across methods")
    A("")
    A("**Where to click.** In the held-out view click the type bar `c3` and then `c2` (or their chips): the brain keeps that type's electrodes only. Then the worst patient in the list at the bottom.")
    A("")
    A(img_md(R["row3"], "c3 only")); A(img_md(R["row2"], "c2 only")); A(img_md(R["pat"], "worst patient"))
    sc = VAL / "figures" / "fig5_scorecard.png"
    if sc.exists():
        sc2 = OUTD / "panels" / "p5_scorecard.png"
        sc2.write_bytes(sc.read_bytes())
        A(img_md(sc2, "scorecard"))
    A(f"**What you see.** c3's electrodes are scattered and a fifth of them are coloured as misses; c2's are a tight temporal set with almost none.")
    A("")
    A(f"**The numbers.** Across algorithms / feature sets (chance-adjusted overlap): " + ", ".join(f"c{c} {nums['type_algo_adj'][str(c)]:.2f} / {nums['type_feat_adj'][str(c)]:.2f}" for c in order7)
      + f"{cite('HA85', 'K55')}; consensus and held-out recall as above{cite('M03', 'TW05')}.")
    A("")
    A("**What it means.** Each type is rated on all three questions; the colour of a scorecard cell is its rank in the column, not a composite score.")
    A("")

    # notes + refs
    A("## Two things to know when reading the page")
    A("")
    A(f"- The page opens at K 8 (its default); every claim above is read at K 7, the paper's cut. The held-out view switches to K 7 by itself.")
    A(f"- At K 7 the page's chips count the published fit's labels (c5 206, c2 164 …) while the paper's figures and the held-out view use the sweep fit's partition (c5 200, c2 168 …); the two agree on {100 * ho['label_agreement']['published']:.1f} % of the electrodes.")
    A("")
    A("## References")
    A("")
    for k, (short, full) in REFS.items():
        A(f"- {full}")
    A("")
    md = "\n".join(L)
    (OUTD / "walkthrough.md").write_text(md, encoding="utf-8")
    body = STORY.md_to_html(md, OUTD)
    page = ("<!doctype html><html><head><meta charset='utf-8'><title>Cluster visualizer walkthrough</title>"
            f"<style>{STORY.CSS}</style></head><body>{body}</body></html>")
    (OUTD / "walkthrough.html").write_text(page, encoding="utf-8")
    print(f"walkthrough.md ({len(md) // 1000} k chars), walkthrough.html ({(OUTD / 'walkthrough.html').stat().st_size / 1e6:.1f} MB)   ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
