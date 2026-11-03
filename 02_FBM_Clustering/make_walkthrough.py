#!/usr/bin/env python3
"""The cluster visualizer, claim by claim: a report that states each claim of Paper 2's
validation story (the types, agreement, stability, separability, the scorecard), names the
paper figure it belongs to, gives the click path that shows it on cluster_visualizer.html,
shows the brain and the statistics that path produces, quotes the numbers from the files,
and ends every statement with the paper it rests on.

    python make_walkthrough.py                  # outputs/clustering/paper2_validation/walkthrough/
    python make_walkthrough.py --reuse-renders  # keep the brain PNGs already there (no VTK)

Figure numbers follow the validation story (Fig 1 types, Fig 2 agreement, Fig 3 stability,
Fig 4 separation, Fig 5 scorecard, Fig 6 LanA), which the manuscript's captions use.

The brain images are rendered here, headless, from the SAME files the page reads
(contacts.json, labels_by_k.json, heldout.json, the figure order) with the page's own
encodings (its type hues by figure position, its patient hues, its loading size and fade,
the held-out view's greyscale and destination colours), on the fsaverage glass brains of
the types figure - so each one is what the page shows after the clicks, at print
resolution. Pane screenshots of the live page, when the builder is given some, are added
beside them.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
REUSE = "--reuse-renders" in sys.argv
NL = chr(10)

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
TEMPORAL = ("temporal", "sts", "heschl", "transverse", "fusiform", "parahippocampal", "hippocampus",
            "amygdala", "insula", "entorhinal")


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
    names = contacts.get("regions") or []
    ridx = [int(r) for r in contacts.get("region", [])]
    region = np.array([names[i] if 0 <= i < len(names) else "" for i in ridx]) if ridx else None
    return dict(cache=cache, man=man, contacts=contacts, order=order, runs=runs, main_id=main_id, ho=ho,
                nums=nums, d=d, xyz=xyz, hemi=hemi, pat_index=pat_index, region=region)


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


def dominated(D, run_id, k):
    """(cluster, patient, n, total) of every cluster more than half one patient, and the largest share overall."""
    st = sweep_stats(D, run_id, k)
    doms, worst = [], (0.0, None, None)
    for c, cnt in st.get("cluster_patient_counts", {}).items():
        tot = sum(cnt.values()); top = max(cnt.items(), key=lambda kv: kv[1])
        if not tot:
            continue
        share = top[1] / tot
        if share > 0.5:
            doms.append((int(c), top[0], top[1], tot))
        if share > worst[0]:
            worst = (share, top[0], int(c))
    return doms, worst


def jaccard_pos1(D, ref_lab, ref_cluster):
    """How many of position #1's contacts each other run shares with the main run's cluster at
    that position: Jaccard of the two hard sets, plus the best-overlapping cluster of that run."""
    ref = set(i for i, L in enumerate(ref_lab) if L == ref_cluster)
    out = {}
    for rid in D["runs"]:
        if rid == D["main_id"] or str(K) not in D["order"].get(rid, {}):
            continue
        lab = sweep_labels(D, rid, K)
        if lab is None:
            continue
        sets = {c: set(i for i, L in enumerate(lab) if L == c) for c in range(K)}
        jac = {c: len(ref & s) / len(ref | s) for c, s in sets.items() if s}
        p1 = D["order"][rid][str(K)][0]
        best = max(jac, key=jac.get)
        out[rid] = dict(pos1=p1, jac=jac.get(p1, 0.0), best=best, best_jac=jac[best], n=len(sets[p1]))
    return out


def anatomy(D, lab, c):
    """Hemisphere split and temporal share of a type, from the page's region field."""
    m = np.array([L == c for L in lab])
    nL = int((D["hemi"][m] == "L").sum()); nR = int((D["hemi"][m] == "R").sum())
    if D["region"] is None:
        return nL, nR, None
    regs = D["region"][m]
    temp = float(np.mean([any(t in r.lower() for t in TEMPORAL) for r in regs])) if len(regs) else None
    return nL, nR, temp


# ---------------------------------------------------------------- rendering
_SCENE_D = None


def render_state(D, side, cols, rads, alphas, zoom=1.30):
    """One view of the contacts whose radius is > 0, on the types figure's glass scene of this side."""
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
    p = REND / f"{name}.png"
    if REUSE and p.exists():
        return p
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
             + "   ·   rendered from the page's data on the glass brains of the types figure (Fig 1)", fontsize=7.5, color=MUTED, ha="left", va="center")
    fig.canvas.draw()
    strip = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGBA")
    plt.close(fig)
    out.alpha_composite(strip.resize((W, 70), Image.LANCZOS), (0, h))
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
    ho = D["ho"]
    chance = D["nums"].get("lopo_predict", {}).get("chance_mean", 0.52)
    fig = plt.figure(figsize=(9.2, 2.1), dpi=170)
    for x, (v, lab) in zip((0.02, 0.36, 0.70), ((ho["lopo"]["overall"], f"leave one patient out: refit on the other {ho['n_patients'] - 1}," + NL + "place the held-out patient"),
                                                 (ho["halves"]["overall"], "patient halves: refit on half the patients," + NL + "place the other half (harsher)"),
                                                 (chance, "chance: every electrode given its" + NL + "patient's commonest type"))):
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
        ax.set_title(ttl + NL + "% of each type's held-out electrodes", fontsize=9, loc="left", color=INK)
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


# ---------------------------------------------------------------- the MMD cells: every feature set x algorithm
MMD_ROOT = ROOT / "outputs" / "clustering" / "mmd"
VARIANT_SETS = {"concat_bands15z": "15 bands z", "concat_bands15s": "15 bands shape", "concat_bands15t": "15 bands ternary"}
FS_SHORT = {"concat_bands15": "15 bands", "concat_bands5": "5 bands", "concat_bands2": "2 bands", "concat_hfa": "HFA", **VARIANT_SETS}
METHOD_SHORT = {"cnmf": "convex NMF", "kmeans": "k-means", "hierarchical": "Ward"}


def mmd_cells(D):
    """Every MMD cell on disk for this cache: the paper's four sets x three methods, then the
    variants (listed as missing until their cells are run)."""
    cells = []
    root = MMD_ROOT / D["cache"]
    for fset in list(P2.FSETS) + list(VARIANT_SETS):
        for method in ("cnmf", "kmeans", "hierarchical"):
            cd = root / fset / method
            if not (cd / "manifest.json").exists() or not (cd / "confusion_lopo_counts.csv").exists():
                if fset in VARIANT_SETS and method != "hierarchical":
                    cells.append(dict(fset=fset, method=method, missing=True))
                continue
            man = json.loads((cd / "manifest.json").read_text(encoding="utf-8"))
            K_ = int(man["k"])
            pt = pd.read_csv(cd / "per_type.csv"); pt = pt[pt.version == "argmax"].set_index("cluster").sort_index()
            ps = pd.read_csv(cd / "pairs_summary.csv"); ps = ps[ps.version == "argmax"].reset_index(drop=True)
            cl = pd.read_csv(cd / "confusion_lopo_counts.csv", index_col=0).to_numpy(float)
            ch = pd.read_csv(cd / "confusion_halves_counts.csv", index_col=0).to_numpy(float)
            rid = f"{method}__{fset}__{man['run']}"
            order = D["order"].get(rid, {}).get(str(K_)) or list(range(K_))
            imin = int(ps.mmd2_ho.idxmin())
            cells.append(dict(fset=fset, method=method, k=K_, run=man["run"], rid=rid, order=[int(x) for x in order],
                              lopo=float(np.trace(cl) / cl.sum()), halves=float(np.trace(ch) / ch.sum()),
                              cl=cl, ch=ch, pt=pt, ps=ps, cloud=float(ps.cloud_median.iloc[0]),
                              below=int((ps.ratio < 1).sum()), n_pairs=int(len(ps)),
                              weakest=int(pt.lopo_recall.idxmin()), weakest_recall=float(pt.lopo_recall.min()),
                              closest=(int(ps.loc[imin, "i"]), int(ps.loc[imin, "j"]), float(ps.loc[imin, "mmd2_ho"]), float(ps.loc[imin, "ratio"])),
                              sil_lo=(int(pt.sil_z.idxmin()), float(pt.sil_z.min())), sil_hi=(int(pt.sil_z.idxmax()), float(pt.sil_z.max())),
                              n_perm=man.get("n_perm"), n_splits=man.get("n_splits")))
    return cells


def panel_cell(c, name, names=None, chance=None):
    """One cell, four panels in the page's colours: A the leave-one-patient-out confusion, B the
    held-out MMD² against the Gaussian cloud, C held-out recall per type under both tests, D the
    silhouette z per type against the cloud's own band."""
    from matplotlib.colors import TwoSlopeNorm
    names = names or {}
    K_, order = c["k"], c["order"]
    hue = {j: u2hue(p, K_) / 255 for p, j in enumerate(order)}
    lab = lambda j: f"c{j}" + (f" {names[j]}" if j in names else "")
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), dpi=150,
                             gridspec_kw=dict(width_ratios=[1, 1.08], height_ratios=[1.15, 1], wspace=0.62, hspace=0.45))
    # A - the confusion
    ax = axes[0, 0]
    R = c["cl"] / np.maximum(c["cl"].sum(1, keepdims=True), 1e-12); Ro = R[np.ix_(order, order)]
    ax.imshow(Ro, cmap="Greys", vmin=0, vmax=1)
    for i in range(K_):
        for j in range(K_):
            ax.text(j, i, f"{100 * Ro[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if Ro[i, j] > 0.55 else INK, fontweight="bold" if i == j else "normal")
    ax.set_title(f"A · leave one patient out: {100 * c['lopo']:.0f} % land in their type" + NL
                 + "% of each published type's held-out electrodes, by where they were placed", fontsize=8, loc="left")
    ax.set_xlabel("placed in", fontsize=7.5, color=MUTED); ax.set_ylabel("published type", fontsize=7.5, color=MUTED)
    # B - held-out MMD² as a ratio to the cloud, diverging around 1
    ax = axes[0, 1]
    M = np.full((K_, K_), np.nan)
    for _, r in c["ps"].iterrows():
        M[int(r.i), int(r.j)] = M[int(r.j), int(r.i)] = r.ratio
    Mo = M[np.ix_(order, order)]
    vmax = max(2.0, float(np.nanmax(Mo)))
    im = ax.imshow(np.where(np.isfinite(Mo), Mo, 1.0), cmap="RdBu", norm=TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=vmax))
    for i in range(K_):
        for j in range(K_):
            if i != j and np.isfinite(Mo[i, j]):
                dark = Mo[i, j] > 1 + 0.72 * (vmax - 1) or Mo[i, j] < 0.4
                ax.text(j, i, f"{Mo[i, j]:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if dark else INK, fontweight="bold" if Mo[i, j] < 1 else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6.5); cb.set_label("held-out MMD² / Gaussian cloud", fontsize=7)
    ax.set_title(f"B · held-out MMD² against the Gaussian cloud: {c['below']} of {c['n_pairs']} pairs below 1" + NL
                 + "red, bold: closer than two sectors of a featureless cloud · blue: farther", fontsize=8, loc="left")
    for ax in axes[0]:
        ax.set_xticks(range(K_)); ax.set_yticks(range(K_))
        ax.set_xticklabels([f"c{j}" for j in order], fontsize=7); ax.set_yticklabels([lab(j) for j in order], fontsize=7)
        for tl, j in zip(ax.get_xticklabels(), order):
            tl.set_color(hue[j])
        for tl, j in zip(ax.get_yticklabels(), order):
            tl.set_color(hue[j])
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
    # C - recall per type, filled = one patient out, open = halves
    ax = axes[1, 0]; y = np.arange(K_)[::-1]; pt = c["pt"]
    for yi, j in zip(y, order):
        lo, ha = float(pt.loc[j, "lopo_recall"]), float(pt.loc[j, "halves_recall"])
        ax.plot([ha, lo], [yi, yi], color=hue[j], lw=2.4, alpha=0.55, solid_capstyle="round")
        ax.plot(lo, yi, "o", color=hue[j], ms=7.5, mec="white", mew=0.8, zorder=3)
        ax.plot(ha, yi, "o", mfc="white", mec=hue[j], ms=7.5, mew=1.6, zorder=3)
        ax.text(max(lo, ha) + 0.025, yi, f"{100 * lo:.0f} % · {100 * ha:.0f} %", va="center", fontsize=7, color=INK)
    if chance is not None:
        ax.axvline(chance, color=MUTED, lw=0.8, ls="--")
        ax.text(chance, K_ - 0.35, f"chance {100 * chance:.0f} %", fontsize=6.5, color=MUTED, ha="center", va="bottom")
    ax.set_yticks(y); ax.set_yticklabels([lab(j) for j in order], fontsize=7)
    for tl, j in zip(ax.get_yticklabels(), order):
        tl.set_color(hue[j])
    ax.set_xlim(0, 1.32); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "25", "50", "75", "100 %"], fontsize=7)
    ax.set_ylim(-0.7, K_ - 0.3 + 0.6)
    ax.set_title("C · held-out recall per type" + NL + "filled: one patient out · open: patient halves", fontsize=8, loc="left")
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=2)
    # D - silhouette z against the cloud
    ax = axes[1, 1]
    z = [float(pt.loc[j, "sil_z"]) for j in order]
    ax.axvspan(-2, 2, color="#ececec", zorder=0); ax.axvline(0, color=MUTED, lw=0.8)
    ax.barh(y, z, color=[hue[j] for j in order], height=0.62, zorder=2)
    for yi, v in zip(y, z):
        ax.text(v + (0.12 if v >= 0 else -0.12), yi, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=7, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([f"c{j}" for j in order], fontsize=7)
    for tl, j in zip(ax.get_yticklabels(), order):
        tl.set_color(hue[j])
    zmax = max(3.5, float(np.nanmax(np.abs(z))) + 1.2); ax.set_xlim(-zmax, zmax)
    ax.tick_params(labelsize=7, length=2); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("D · silhouette against the Gaussian cloud, as z" + NL + "grey band: within two SD of the cloud's own silhouette", fontsize=8, loc="left")
    fig.suptitle(f"{FS_SHORT.get(c['fset'], c['fset'])} × {METHOD_SHORT[c['method']]}   ·   K {K_}   ·   run {c['run']}   ·   clusters in the page's figure order and colours",
                 fontsize=9.5, x=0.02, ha="left", y=0.995)
    p = OUTD / "panels" / f"{name}.png"
    fig.savefig(p, facecolor="white", bbox_inches="tight"); plt.close(fig)
    return p


# ---------------------------------------------------------------- the report
def img_md(p, alt):
    return f"![{alt}](" + str(Path(p).relative_to(OUTD)).replace(chr(92), "/") + ")" + NL


def pct(v):
    return f"{100 * float(v):.0f} %"


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
    lab7 = sweep_labels(D, mid, 7)
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
        pos1[m_] = (rid, D["order"][rid]["7"][0])
    R["algo"] = [state_type(D, rid, 7, c1, 0, 7, f"04_pos1_{m_}", f"position #1 under {P2.METHOD_LABEL[m_]} on 15 bands at K 7 = cluster {c1}") for m_, (rid, c1) in pos1.items()]
    fs1 = {}
    for f_ in P2.FSETS:
        rid = next(i for i in D["runs"] if i.startswith(f"cnmf__{f_}__"))
        fs1[f_] = (rid, D["order"][rid]["7"][0])
    R["fset"] = [state_type(D, rid, 7, c1, 0, 7, f"05_pos1_{f_}", f"position #1 for convex NMF on {P2.FS_LABEL[f_]} at K 7 = cluster {c1}") for f_, (rid, c1) in fs1.items()]
    J = jaccard_pos1(D, lab7, order7[0])
    # stability: the K 8 cluster one patient owns, and the largest share at K 7
    doms8, _ = dominated(D, mid, 8)
    _, worst7 = dominated(D, mid, 7)
    dom = doms8[0] if doms8 else None
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
    pub = Counter(int(v[0]) for v in D["contacts"]["runs"][mid] if v is not None)
    swp = Counter(L for L in lab7 if L is not None)
    chance = nums.get("lopo_predict", {}).get("chance_mean", 0.52)
    always = sum(1 for v in ho["contacts"]["lopo_hit"] if v is not None and v >= 1)
    never = sum(1 for v in ho["contacts"]["lopo_hit"] if v is not None and v <= 0)

    # ---- the text
    L = []
    A = L.append
    A("# The cluster visualizer, claim by claim")
    A("")
    A(f"*Built {time.strftime('%Y-%m-%d %H:%M')} from the files the page reads (cohort {D['cache']}, run {D['runs'][mid]['run']}). "
      f"Page: {PAGE}. The brain images are rendered from the same data with the page's own encodings, on the glass brains of the "
      f"types figure (Fig 1); the click path above each one reproduces it live.*")
    A("")
    A("## What this report is")
    A("")
    A("One claim per section, five claims: the types, their agreement across descriptions and algorithms, their stability across "
      "patients, their separability, and the scorecard that puts the three together. Each section names the paper figure the claim "
      "belongs to, gives the clicks that show it on the cluster visualizer, shows the brain and the statistics those clicks produce, "
      "reads the numbers off the files, and says what they mean, with the paper each measure rests on.")
    A("")
    A("## Figure numbers")
    A("")
    A("Numbers follow the validation story, which the manuscript's captions already use (Fig 2C, 2D, 3A). The paper's new opening "
      "figure, the task with one exemplar electrode per type, has no number in the story; if it becomes Figure 1, every number below "
      "moves up by one.")
    A("")
    A("| claim | question | figure | panels |")
    A("|---|---|---|---|")
    A("| 1 | what the seven types are | Fig 1, the types | mean ERSP per type, the brains, the patient composition; the blocks' order is the chips' order |")
    A("| 2 | agreement | Fig 2 | A, B the ARI and NMI matrices with the modal-agreement map; C the matched clusters against the two nulls (was FIG 4); D the 66-pair diagonal share (was FIG 5) |")
    A("| 3 | stability | Fig 3 | A one-patient clusters over K (was FIG 1 D); B–D the consensus matrix, the per-type consensus, PAC against the Gaussian cloud; E leave one patient out; F how much of K 7 survives K 8, 9, 10. The composite draft letters them differently: B = K 7 → 8, C–E consensus, F = leave one out |")
    A("| 4 | separation | Fig 4 | A silhouette z per type; B where held-out electrodes land; C the held-out MMD matrix; D the most and least separated pairs; E the loadings; F the patient-halves confusion; G the per-electrode hit rate |")
    A("| 5 | all three at once | Fig 5 | the scorecard |")
    A("")
    A("## The page's controls, in its own words")
    A("")
    A("| control | where | what it does |")
    A("|---|---|---|")
    A("| `Run` | the feature set × algorithm matrix at the top of the control panel | picks the run; switching keeps the chip position, and the run note prints the ARI against the run you came from |")
    A("| `Clusters (K)` | the row of numbers under Run | picks K; the page opens at K 8 |")
    A("| chips `#1 … #7` | under Cluster | the clusters in figure order: #1 is the first block of the types figure (Fig 1). A chip shows n, the biggest patient and its share, and a red `*` when one patient holds more than half |")
    A("| `View` | one · overlay · grid · held-out | one cluster; several at once; the same position across runs (`algorithms` or `feature sets`); or the held-out panel |")
    A("| held-out panel | right of the brain | six numbered sections: 1 the answer in one number; 2 per type (click a bar); 3 where the misses go (click a cell); 4 on the brain (colour by, source, patient); 5 is it patients? (click a patient); 6 what it means |")
    A("| `Minimum loading` | bottom of the control panel | hides electrodes below a loading; the counts and the red `*` are recomputed |")
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
    A("## Claim 1 · The clustering gives seven recurring response types, each carried by many patients")
    A("")
    A("**Paper figure.** Fig 1, the types. Its seven blocks are the chips `#1 … #7`, in that order.")
    A("")
    A("**Where to click.**")
    A("")
    A("- **1.** `Run`: 15 bands × cNMF.")
    A("- **2.** `Clusters (K)`: 7.")
    A("- **3.** `View` one, chip `#1` (c5 production); then chip `#7` (c3 sentence-stimulus low-frequency).")
    A("- **4.** All seven at once: `View` overlay, then click every chip.")
    A("")
    A(img_md(R["c5"], "c5")); A(img_md(R["c3"], "c3")); A(img_md(R["all"], "all types"))
    A("**What you see.** Every dot is one electrode of the type. Its size and its fade carry its loading on the type: small and grey "
      "below 0.5. The chip carries n and the biggest patient's share; the red `*` that marks a patient holding more than half of a "
      "cluster is absent at K 7.")
    A("")
    A("**The numbers.**")
    A("")
    A("| type | n | patients | largest patient's share |")
    A("|---|---|---|---|")
    for c in order7:
        A(f"| {nm[c]} | {T[c]['n']} | {T[c]['n_patients']} | {pct(nums['top_share'][str(c)])} |")
    A("")
    A(f"Median own-type loading {nums['own_median']:.2f}; {pct(nums['own_below_cut'])} of the electrodes sit below the {P2.LOAD_CUT:.1f} draw cut{cite('D10')}.")
    A("")
    A(f"**What it means.** Seven shapes, each spread over {min(t['n_patients'] for t in T.values())} to {max(t['n_patients'] for t in T.values())} patients. "
      "The mixed loadings are the first sign that the types are regions of a continuum rather than islands, which Claim 4 measures" + cite("D10") + ".")
    A("")

    # claim 2
    aa, af = nums["ari_algo"], nums["ari_feat"]
    rk = next(i for i in D["runs"] if i.startswith(f"kmeans__{FSET}__")); rw = next(i for i in D["runs"] if i.startswith(f"hierarchical__{FSET}__"))
    jk, jw = J[rk], J[rw]
    jf = {f_: J[next(i for i in D["runs"] if i.startswith(f"cnmf__{f_}__"))] for f_ in P2.FSETS if f_ != FSET}
    A("## Claim 2 · Other algorithms find the same shapes; the frequency description matters more than the algorithm")
    A("")
    A("**Paper figure.** Fig 2, agreement: A and B the ARI and NMI matrices with the modal-agreement map, C the matched clusters against the two nulls, D the diagonal share.")
    A("")
    A("**Where to click.**")
    A("")
    A("- **1.** `View` grid, `algorithms`: chip `#1` under convex NMF, k-means and Ward at K 7. In k-means and Ward, position #1 is the cluster paired "
      "one-to-one with convex NMF's production cluster, by Hungarian assignment on the electrodes two clusters share" + cite("K55") + ".")
    A("- **2.** `feature sets`: chip `#1` on 15, 5, 2 bands and HFA. Here every run's chips follow its own types figure (Fig 1, and S3 for the other three sets), so position #1 is each feature set's first block.")
    A("- **3.** Switch a run in `Run`: the position is kept and the run note prints the ARI against the run you came from.")
    A("")
    for p in R["algo"]: A(img_md(p, "position 1 across algorithms"))
    for p in R["fset"]: A(img_md(p, "position 1 across feature sets"))
    A(img_md(Pn["ari"], "ARI"))
    A("**What you see.** k-means puts the production type on much the same contacts as convex NMF; Ward's partner is a different set of contacts, "
      "and the Ward cluster that overlaps production most sits at another position. Across feature sets the first block is production-like "
      "each time, on about a third of the same contacts.")
    A("")
    A("**The numbers.**")
    A("")
    A("| comparison | ARI | contacts shared by position #1 (Jaccard with c5) |")
    A("|---|---|---|")
    A(f"| k-means vs convex NMF | {aa['k-means vs convex NMF']['ari']:.2f} | {jk['jac']:.2f} |")
    A(f"| k-means vs Ward | {aa['k-means vs Ward']['ari']:.2f} | – |")
    A(f"| Ward vs convex NMF | {aa['Ward vs convex NMF']['ari']:.2f} | {jw['jac']:.2f} (c5's best-overlapping Ward cluster: {jw['best_jac']:.2f}, at position #{D['order'][rw]['7'].index(jw['best']) + 1}) |")
    for f_, lbl in (("concat_bands5", "15 bands vs 5 bands"), ("concat_bands2", "15 bands vs 2 bands"), ("concat_hfa", "15 bands vs HFA")):
        A(f"| {lbl} | {af[lbl]['ari']:.2f} | {jf[f_]['jac']:.2f} |")
    for lbl in ("5 bands vs 2 bands", "5 bands vs HFA", "2 bands vs HFA"):
        A(f"| {lbl} | {af[lbl]['ari']:.2f} | – |")
    A("")
    A(f"All three algorithms agree on {pct(nums['modal_algo']['3'])} of the electrodes, all four feature sets on {pct(nums['modal_feat']['4'])}{cite('HA85')}.")
    A("")
    A("**What it means.** Changing the algorithm moves the electrodes less than changing the frequency description" + cite("HA85") + ". "
      "Agreement is not correctness: four descriptions can agree on an artefact of the gate.")
    A("")

    # claim 3
    pac = nums["pac"]
    cons_rank = sorted(T, key=lambda c: -(T[c]["consensus"] or 0))
    A("## Claim 3 · The shapes come back when patients are resampled, and K 7 is the last K before one patient owns a cluster")
    A("")
    A("**Paper figure.** Fig 3, stability: A the one-patient clusters over K, B–D the consensus panels, E leave one patient out, F K 7 → 8.")
    A("")
    A("**Where to click.**")
    A("")
    A("- **1.** `Clusters (K)`: 8. One chip now carries the red `*` (a patient holds more than half of it); back at 7 it is gone.")
    A("- **2.** `View` held-out → section 4 · On the brain → colour by `consensus margin`.")
    A("- **3.** Source `patient halves`; section 5 · Is it patients? lists the patients, worst first.")
    A("")
    if dom:
        A(img_md(R["k8"], "K 8 one-patient cluster"))
    A(img_md(R["cons"], "consensus margin")); A(img_md(R["miss_h"], "misses under the halves test")); A(img_md(Pn["pats"], "patients"))
    A("**What you see.** At K 8 the extra cluster is one patient's strip of contacts; at K 7 no cluster is more than half one patient. "
      "Under consensus margin the dark electrodes stay with their type when 20 % of the patients are dropped and the light ones switch. "
      "Under the halves test the misses concentrate in a few patients.")
    A("")
    A("**The numbers.**")
    A("")
    if dom:
        A(f"- K 8: cluster {dom[0]} is {pct(dom[2] / dom[3])} one patient ({dom[1]}, {dom[2]} of {dom[3]}). K 7: the largest share is {pct(worst7[0])} ({worst7[1]} in {nm[worst7[2]]}).")
    A(f"- PAC {pac['real']:.3f} against {pac['null']:.3f} ± {pac['null_sd']:.3f} for the Gaussian cloud{cite('M03', 'S14')}.")
    A(f"- The halves placement keeps {pct(ho['halves']['overall'])} overall; " + ", ".join(f"{r['p']} {r['excess']:.1f}×" for r in worst3)
      + f" carry more than their share of the misses; {keep80} of {len(pats)} patients {'keeps' if keep80 == 1 else 'keep'} at least 80 %{cite('L04', 'V17')}.")
    A("")
    A("| type | consensus (mean M within the type) |")
    A("|---|---|")
    for c in cons_rank:
        A(f"| {nm[c]} | {T[c]['consensus']:.2f} |")
    A("")
    A(f"Consensus is the share of resampling runs in which two electrodes land together{cite('M03')}.")
    A("")
    A("**What it means.** The partition is stable against patients, and K 7 is where that stops. The cloud has no patient structure, so the dominance rule is the guard" + cite("S14") + ".")
    A("")

    # claim 4
    rec = {c: T[c]["lopo_recall"] for c in T}
    A("## Claim 4 · The types are separable: an electrode from an unseen patient lands in its type nine times in ten, and the leaks run between neighbours")
    A("")
    A("**Paper figure.** Fig 4, separation: B is this test (where held-out electrodes land), F its patient-halves version, G the per-electrode hit rate; A, C and D are the silhouette and MMD readings of the same question.")
    A("")
    A("**Where to click.**")
    A("")
    A("- **1.** `View` held-out, source `leave one patient out`, colour by `where misses go`. Section 1 gives the three numbers, section 2 the per-type bars.")
    A("- **2.** Section 3: click the cell `c3 → c4` in matrix B (patient halves). The brain lights the electrodes of that cell and steps the rest back.")
    A("- **3.** Section 4: colour by `hit rate`.")
    A("")
    A(img_md(Pn["numbers"], "one number")); A(img_md(Pn["types"], "per type")); A(img_md(R["miss_l"], "misses LOPO")); A(img_md(Pn["conf"], "confusions")); A(img_md(R["cell"], "cell c3 to c4")); A(img_md(R["hit"], "hit rate"))
    A("**What you see.** Most electrodes are small grey hits. The coloured dots are misses, painted in the colour of the type they were placed in; "
      "for c3 they are spread over both hemispheres and many patients, so the leak is neither one patient nor one place. The clicked cell shows "
      "exactly which c3 electrodes the halves test puts in c4.")
    A("")
    A("**The numbers.**")
    A("")
    A(f"{pct(ho['lopo']['overall'])} of held-out electrodes land in their published type with one patient out, {pct(ho['halves']['overall'])} with half the patients out, "
      f"chance about {pct(chance)}{cite('TW05', 'L04')}.")
    A("")
    A("| type | leave one patient out | patient halves | loses most to (one out) | loses most to (halves) |")
    A("|---|---|---|---|---|")
    for c in sorted(rec, key=rec.get, reverse=True):
        t = T[c]
        A(f"| {nm[c]} | {pct(t['lopo_recall'])} | {pct(t['halves_recall'])} | c{t['lopo_confused_with']} ({pct(t['lopo_confused_share'])}) | c{t['halves_confused_with']} ({pct(t['halves_confused_share'])}) |")
    A("")
    A(f"Per electrode, {always} of {ho['n']} land in their type in every fold and {never} never do; the refit clusters are aligned to the published ones by Hungarian assignment before counting{cite('TW05', 'K55')}.")
    A("")
    A("**What it means.** Separable in the sense that matters, an unseen electrode can be told apart, and only partly: the leaky types lose a fifth of their electrodes to their neighbours, which is what a continuum of responses looks like under a partition" + cite("TW05") + ".")
    A("")

    # claim 4 on every cell: the MMD run's tables
    cells = mmd_cells(D)
    have = [c for c in cells if not c.get("missing")]
    miss = [c for c in cells if c.get("missing")]
    A("## Claim 4 on every feature set and algorithm · the separation tables of the MMD run")
    A("")
    A("**Paper figure.** Fig 4 C and D (the held-out MMD against the Gaussian cloud) and S5; the placement test of Claim 4 (Fig 4 B) is run on every cell the same way.")
    A("")
    A("**Where to click.** `Run`: pick the cell in the matrix, then `View` held-out. A run whose tables are published shows the panel; the others say that none are published yet. "
      "The cells below are the MMD run's own tables on disk, which is where the page's numbers come from.")
    A("")
    for c in have:
        if c["method"] == "cnmf":
            main_cell = (c["fset"] == FSET and c["method"] == METHOD)
            A(img_md(panel_cell(c, f"p6_cell_{c['fset']}_{c['method']}", names=({k_: T[k_]["name"] for k_ in T} if main_cell else None),
                                chance=(chance if main_cell else None)), f"{FS_SHORT.get(c['fset'], c['fset'])} x convex NMF"))
    A("**What you see.** One figure per feature set, convex NMF, clusters in the page's order and colours: A the leave-one-patient-out confusion, "
      "B every pair's held-out MMD² as a ratio to the cloud (red and bold below 1: the pair is closer than two sectors of a featureless Gaussian), "
      "C each type's held-out recall under both tests, D its silhouette z against the cloud (the grey band is the cloud's own spread). "
      "k-means and Ward have the same tables; the summary below carries every cell.")
    A("")
    A("**The numbers.**")
    A("")
    A("| feature set | algorithm | K | one patient out | patient halves | weakest type | closest pair · MMD² · ratio to cloud | pairs below the cloud | silhouette z, lowest … highest |")
    A("|---|---|---|---|---|---|---|---|---|")
    for c in have:
        A(f"| {FS_SHORT.get(c['fset'], c['fset'])} | {METHOD_SHORT[c['method']]} | {c['k']} | {pct(c['lopo'])} | {pct(c['halves'])} | c{c['weakest']} ({pct(c['weakest_recall'])}) | "
          f"c{c['closest'][0]}–c{c['closest'][1]} · {c['closest'][2]:.3f} · {c['closest'][3]:.2f} | {c['below']} of {c['n_pairs']} | c{c['sil_lo'][0]} {c['sil_lo'][1]:+.1f} … c{c['sil_hi'][0]} {c['sil_hi'][1]:+.1f} |")
    for c in miss:
        A(f"| {FS_SHORT.get(c['fset'], c['fset'])} | {METHOD_SHORT[c['method']]} | 7 | not run yet | | | | | |")
    A("")
    if have:
        lo = min(have, key=lambda c: c["lopo"]); hi = max(have, key=lambda c: c["lopo"])
        by_cn = [c for c in have if c["method"] == "cnmf"]
        A(f"Leave-one-patient-out placement runs from {pct(lo['lopo'])} ({FS_SHORT.get(lo['fset'], lo['fset'])} × {METHOD_SHORT[lo['method']]}) to {pct(hi['lopo'])} "
          f"({FS_SHORT.get(hi['fset'], hi['fset'])} × {METHOD_SHORT[hi['method']]}); the patient-halves test is harsher everywhere{cite('TW05', 'L04')}. "
          f"Pairs below the cloud, convex NMF: " + ", ".join(f"{FS_SHORT.get(c['fset'], c['fset'])} {c['below']} of {c['n_pairs']}" for c in by_cn) + f"{cite('G12')}. "
          f"The bandwidth, the cloud and the shuffles are those of the MMD run (n_perm {have[0]['n_perm']}, splits {have[0]['n_splits']}){cite('G12', 'R87')}.")
        A("")
    if miss:
        A("The variant cells are listed as not run yet: on the server, `python 00_paper2_figure6_mmd.py --sets concat_bands15z concat_bands15s --methods cnmf kmeans` "
          "(the main cell's settings: `--n-perm 5000 --n-perm-heldout 500 --n-splits 100 --n-cloud 100`), then `python make_heldout_bundle.py --write` and this report again.")
        A("")
    A("**What it means.** The separation reading is the same in every description: most held-out electrodes land in their type, and the pairs that sit below the cloud are the neighbours of a continuum, not distinct islands"
      + cite("TW05", "G12") + ". K differs between sets (the one-patient rule picks it per set), so the pair counts are read within a set, not across.")
    A("")


    # claim 5
    an = {c: anatomy(D, lab7, c) for c in (2, 3)}
    cohort_temp = float(np.mean([any(t in r.lower() for t in TEMPORAL) for r, Lc in zip(D["region"], lab7) if Lc is not None])) if D["region"] is not None else None
    A("## Claim 5 · The scorecard: c2 and c4 solid, c1 and c3 fragile, c5 solid under patients but weak across methods")
    A("")
    A("**Paper figure.** Fig 5, the scorecard.")
    A("")
    A("**Where to click.**")
    A("")
    A("- **1.** In the held-out view click the type bar `c3` in section 2, then `c2`: the brain keeps that type's electrodes only.")
    A("- **2.** Section 5: click the worst patient.")
    A("")
    A(img_md(R["row3"], "c3 only")); A(img_md(R["row2"], "c2 only")); A(img_md(R["pat"], "worst patient"))
    sc = VAL / "figures" / "fig5_scorecard.png"
    if sc.exists():
        sc2 = OUTD / "panels" / "p5_scorecard.png"
        sc2.write_bytes(sc.read_bytes())
        A(img_md(sc2, "scorecard"))
    tp3 = f"{pct(an[3][2])} in temporal cortex against {pct(cohort_temp)} for the whole cohort" if an[3][2] is not None else "over both hemispheres"
    tp2 = f"{pct(an[2][2])} temporal" if an[2][2] is not None else "mostly temporal"
    A(f"**What you see.** c3's electrodes are spread over both hemispheres ({an[3][0]} left, {an[3][1]} right; {tp3}) and a fifth of them are coloured as misses. "
      f"c2's are {tp2} ({an[2][0]} left, {an[2][1]} right) with almost none coloured.")
    A("")
    A("**The numbers.** The scorecard's columns, one row per type:")
    A("")
    A("| type | overlap across algorithms | overlap across feature sets | consensus | one patient out | patient halves | silhouette z | patients | largest share |")
    A("|---|---|---|---|---|---|---|---|---|")
    for c in order7:
        t = T[c]
        A(f"| {nm[c]} | {nums['type_algo_adj'][str(c)]:.2f} | {nums['type_feat_adj'][str(c)]:.2f} | {t['consensus']:.2f} | {pct(t['lopo_recall'])} | {pct(t['halves_recall'])} | {t['sil_z']:+.1f} | {t['n_patients']} | {pct(nums['top_share'][str(c)])} |")
    A("")
    A(f"Overlaps are chance-adjusted Jaccards of matched clusters{cite('HA85', 'K55')}; consensus as in Claim 3{cite('M03')}; the held-out columns as in Claim 4{cite('TW05')}; "
      f"silhouette z is the type's silhouette against the Gaussian cloud{cite('R87')}.")
    A("")
    A("**What it means.** Each type is rated on all three questions; the colour of a scorecard cell is its rank in the column, not a composite score.")
    A("")

    # notes + refs
    A("## Two things to know when reading the page")
    A("")
    A("- The page opens at K 8 (its default); every claim above is read at K 7, the paper's cut. The held-out view switches to K 7 by itself.")
    A(f"- At K 7 the page's chips count the published fit's labels (c5 {pub[5]}, c2 {pub[2]}, c4 {pub[4]} …) while the paper's figures and the held-out view use the "
      f"sweep fit's partition (c5 {swp[5]}, c2 {swp[2]}, c4 {swp[4]} …); the two agree on {100 * ho['label_agreement']['published']:.1f} % of the electrodes.")
    A("")
    A("## References")
    A("")
    for k, (short, full) in REFS.items():
        A(f"- {full}")
    A("")
    md = NL.join(L)
    (OUTD / "walkthrough.md").write_text(md, encoding="utf-8")
    body = STORY.md_to_html(md, OUTD)
    page = ("<!doctype html><html><head><meta charset='utf-8'><title>Cluster visualizer walkthrough</title>"
            f"<style>{STORY.CSS}</style></head><body>{body}</body></html>")
    (OUTD / "walkthrough.html").write_text(page, encoding="utf-8")
    print(f"walkthrough.md ({len(md) // 1000} k chars), walkthrough.html ({(OUTD / 'walkthrough.html').stat().st_size / 1e6:.1f} MB)   ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
