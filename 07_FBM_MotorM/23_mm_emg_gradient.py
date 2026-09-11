"""Step 4 - is the residual mouth contamination shaped like a source OUTSIDE the skull?

    python .\\23_mm_emg_gradient.py --patient G-05 --run wm
    python .\\23_mm_emg_gradient.py --patient G-05 --run none

THE QUESTION. After WM referencing, mouth-minus-foot is still +0.3 dB on the average
contact and +0.8 to +1.1 dB on temporal shafts. The claim is that this is jaw and
temporalis EMG conducted through the head. That claim makes predictions a neural or
instrumental explanation does not have to satisfy, and each one below is a test of one
of them.

FOUR TESTS.

  1. DEPTH ALONG THE SHAFT, IN MILLIMETRES. A source outside the skull must be weaker the
     deeper you go - that is what volume conduction is, not a modelling choice. Depth is
     measured as the distance along the shaft from its most superficial contact, from the
     fsaverage coordinates, so the result is a SLOPE IN dB PER mm: the number a per-contact
     correction would actually use. Compared within a shaft, everything but depth is held
     constant - same wire, same amplifier input, same reference - which is why this is the
     cleanest test. Reported per shaft as a rank correlation (robust to a few extreme
     contacts) and pooled as a bootstrapped slope.

  2. TISSUE AT MATCHED DEPTH (the none run only, where WM contacts are kept). EMG does not
     care what tissue a contact sits in, only where it is; a neural response does. So:
     take out what depth explains, and ask whether white-matter and grey-matter contacts
     still differ. If they do not, the residual is not neural. This is the strongest single
     discriminator here, and it needs the unreferenced run because the WM run has consumed
     the WM contacts as its reference.

  3. ENTRY CONTACTS, TEMPORAL VERSUS OTHER. The most superficial contact of each shaft is
     the one nearest the muscle. If temporalis is the source, shafts entering through the
     temporal bone should be hottest at the surface and frontal entries cooler. This asks
     "is it temporalis specifically?" directly, using one contact per shaft so no shaft is
     over-weighted by its length.

  4. LATERALITY, |x|. Kept as a picture only. On a lateral shaft "more lateral" and "more
     superficial" are the same contacts, so this is not independent of test 1 and is not
     given a verdict.

Which end of a shaft is the tip is MEASURED from the coordinates and reported, not assumed.

WHAT THE ANSWER LOOKS LIKE. Test 1 positive, test 2 showing no tissue effect, test 3
showing temporal entries hottest: muscle, from outside, temporalis in particular, and a
per-contact quantity that depth predicts - correctable rather than excludable. Test 1 flat
or test 2 showing a clear GM > WM difference: the residual is not organised the way an
outside source must be, and the muscle reading is wrong or incomplete.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "functions"))
import config_mm as cfg                                        # noqa: E402

RECON = os.path.join(ROOT, "..", "02_FBM_Clustering", "outputs", "250_recon", "fsaverage",
                     "coords", "ALL_PATIENTS_contacts_fsaverage.csv")
INK, MUTED, GREY = "#1b232c", "#68727d", "#c9ced4"
RED, BLUE, GREEN = "#c1121f", "#4a6fa5", "#1b7837"
TEMPORAL_WORDS = ("temporal", "hippocamp", "amygdala", "entorhinal", "fusiform",
                  "parahippocampal", "temporalpole")


def shaft_and_index(name):
    m = re.match(r"^(.*?)(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, np.nan)


def temporal_flag(pre, names):
    hits = glob.glob(pre.get("electrodes_tsv", ""))
    if not hits:
        return {n: False for n in names}
    e = pd.read_csv(hits[0], sep="\t")
    lab = {str(r["name"]): str(r.get("tissueLabel", "")).lower() for r in e.to_dict("records")}
    return {n: any(w in lab.get(n, "") for w in TEMPORAL_WORDS) for n in names}


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan, np.nan, int(m.sum())
    r, p = spearmanr(x[m], y[m])
    return float(r), float(p), int(m.sum())


def fmt(r, p, n):
    if not np.isfinite(r):
        return f"n={n:3d}   (too few)"
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"n={n:3d}   rho {r:+.2f}   p {p:.3g} {star}"


def depth_mm_along_shaft(g):
    """Distance along the shaft from its most superficial contact, in mm.

    Cumulative distance between consecutive contacts in fsaverage space, measured from the
    end the recon says is superficial. Sub-mm error from fsaverage warping is irrelevant at
    the scale of a 3.5 mm contact pitch.
    """
    g = g.sort_values("idx")
    P = g[["fx", "fy", "fz"]].to_numpy(float)
    if len(P) < 2 or not np.isfinite(P).all():
        return pd.Series(np.nan, index=g.index)
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(step)])          # from the lowest index
    return pd.Series(cum, index=g.index)


def boot_slope(x, y, n=2000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 6:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = np.polyfit(x, y, 1)[0]
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0:
            bs.append(np.polyfit(x[i], y[i], 1)[0])
    return float(b), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def run(pat, which):
    pre = cfg.MM_PRESETS[pat]
    pid = pre["pat_name"]
    p = os.path.join(ROOT, "outputs", pat, "macro", which, f"{pid}_MM_broadband.tsv")
    if not os.path.exists(p):
        raise SystemExit(f"no {which} run at {p}\n  run:  python .\\20_mm_macro.py "
                         f"--patient {pat} --reref {which}")
    d = pd.read_csv(p, sep="\t")
    d[["shaft", "idx"]] = pd.DataFrame([shaft_and_index(c) for c in d.channel], index=d.index)
    d["y"] = d["mouth_minus_foot"].astype(float)
    d["is_wm"] = d.get("tissue", pd.Series("gm", index=d.index)).astype(str).eq("wm")

    rc = pd.read_csv(os.path.abspath(RECON))
    rc = rc[rc.patient == pid][["name", "x", "y", "z"]].rename(
        columns={"x": "fx", "y": "fy", "z": "fz"})
    d = d.merge(rc, left_on="channel", right_on="name", how="left").drop(columns=["name"])
    d["absx"] = d.fx.abs()
    d["temporal"] = d.channel.map(temporal_flag(pre, d.channel.tolist()))
    print(f"\n=== {pat} ({pid})   run: {which}   {len(d)} contacts, "
          f"{int(d.fx.notna().sum())} with coordinates, {int(d.temporal.sum())} temporal, "
          f"{int(d.is_wm.sum())} white matter")

    # ---- which end is the tip: measured -------------------------------------------
    d["r_centre"] = np.sqrt(d.fx ** 2 + d.fy ** 2 + d.fz ** 2)
    dirs = [rho(g.idx, g.r_centre)[0] for _, g in d.groupby("shaft") if g.fx.notna().sum() >= 3]
    outward = float(np.nanmedian(dirs)) > 0 if dirs else True
    print(f"  contact number vs distance from centre, median rho over shafts "
          f"{np.nanmedian(dirs):+.2f} -> higher numbers are "
          f"{'MORE SUPERFICIAL' if outward else 'DEEPER'}")

    # depth in mm from the superficial end of each shaft
    d["along"] = pd.concat([depth_mm_along_shaft(g) for _, g in d.groupby("shaft")])
    d["depth_mm"] = d.groupby("shaft").along.transform(lambda s: (s.max() - s) if outward else s)
    # depth_mm = 0 at the most superficial contact, growing inward

    # ---- 1. depth along the shaft ---------------------------------------------------
    print("\n  1. mouth-foot against depth along the shaft, mm from the most superficial contact")
    print("     (rho NEGATIVE = contamination falls with depth = what an outside source does)")
    rows = []
    for sh, g in d.groupby("shaft"):
        if g.depth_mm.notna().sum() >= 4:
            rr, pp, nn = rho(g.depth_mm, g.y)
            rows.append((sh, rr, pp, nn, bool(g.temporal.iloc[0]), float(g.depth_mm.max())))
    for sh, rr, pp, nn, tf, L in sorted(rows, key=lambda t: np.nan_to_num(t[1], nan=9)):
        print(f"       {sh:5s} {'temporal' if tf else '        '}  span {L:4.0f} mm  {fmt(rr, pp, nn)}")
    for lab, m in (("temporal", d.temporal), ("other", ~d.temporal),
                   ("all", pd.Series(True, index=d.index))):
        b, (lo, hi) = boot_slope(d.depth_mm[m], d.y[m])
        print(f"     pooled {lab:9s} {fmt(*rho(d.depth_mm[m], d.y[m]))}   "
              f"slope {b*10:+.2f} dB per 10 mm  [95% {lo*10:+.2f}, {hi*10:+.2f}]")

    # ---- 2. tissue at matched depth -----------------------------------------------
    print("\n  2. white matter vs grey matter AFTER depth is taken out")
    if d.is_wm.any() and (~d.is_wm).any():
        m = d.depth_mm.notna() & d.y.notna()
        b = np.polyfit(d.depth_mm[m], d.y[m], 1)
        resid = d.y - np.polyval(b, d.depth_mm)
        w, g_ = resid[d.is_wm & m], resid[~d.is_wm & m]
        U, pv = mannwhitneyu(w, g_, alternative="two-sided")
        print(f"     residual after depth:  WM {w.mean():+.2f} dB (n={len(w)})   "
              f"GM {g_.mean():+.2f} dB (n={len(g_)})   difference {g_.mean()-w.mean():+.2f}   "
              f"Mann-Whitney p {pv:.3g}")
        print("     (no difference = tissue does not matter = not neural; "
              "GM clearly above WM = a cortical component is present)")
    else:
        print("     needs the none run - the WM run has consumed the WM contacts as its reference")

    # ---- 3. entry contacts, temporal vs other --------------------------------------
    print("\n  3. the most superficial contact of each shaft: temporal entries vs the rest")
    # a shaft left with one contact has no along-shaft distance, so its depth is NaN and
    # idxmin would return NaN; drop those before choosing each shaft's surface contact
    _ok = d.dropna(subset=["depth_mm", "y"])
    top = _ok.loc[_ok.groupby("shaft").depth_mm.idxmin()]
    te, ot = top[top.temporal].y, top[~top.temporal].y
    if len(te) >= 2 and len(ot) >= 2:
        U, pv = mannwhitneyu(te, ot, alternative="two-sided")
        print(f"     temporal entries (n={len(te)}): mean {te.mean():+.2f} dB   "
              f"other entries (n={len(ot)}): mean {ot.mean():+.2f} dB   "
              f"difference {te.mean()-ot.mean():+.2f}   p {pv:.3g}")
    for _, r in top.sort_values("y", ascending=False).iterrows():
        print(f"       {r.channel:7s} {'temporal' if r.temporal else '        '}  {r.y:+.2f} dB")

    # ---- 4. laterality, picture only ----------------------------------------------
    print("\n  4. |x| laterality - drawn, not judged: confounded with depth on lateral shafts")
    print(f"     all   {fmt(*rho(d.absx, d.y))}")

    # ---- figure ---------------------------------------------------------------------
    out = os.path.join(ROOT, "outputs", pat, "macro", which, f"{pid}_MM_emg_gradient.png")
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2), dpi=150)
    col = np.where(d.temporal, RED, BLUE)

    ax = axes[0]
    for sh, g in d.groupby("shaft"):
        g = g.sort_values("depth_mm")
        c = RED if g.temporal.iloc[0] else BLUE
        ax.plot(g.depth_mm, g.y, "-", lw=0.7, color=c, alpha=0.5)
        ax.plot(g.depth_mm, g.y, "o", ms=3, color=c, alpha=0.85)
        ax.text(g.depth_mm.iloc[0] - 1, g.y.iloc[0], sh, fontsize=5.5, color=MUTED,
                ha="right", va="center")
    ax.set_xlabel("depth along shaft (mm from most superficial contact)", fontsize=7.5,
                  color=MUTED)
    ax.set_ylabel("mouth − foot (dB)", fontsize=8, color=MUTED)
    ax.set_title("1  along each shaft   (red = temporal)", fontsize=9, color=INK, loc="left")

    ax = axes[1]
    if d.is_wm.any():
        for flag, c, lab in ((False, BLUE, "grey matter"), (True, GREEN, "white matter")):
            m = d.is_wm == flag
            ax.scatter(d.depth_mm[m], d.y[m], s=14, color=c, alpha=0.8, lw=0, label=lab)
        ax.legend(fontsize=6.5, frameon=False)
        ax.set_title("2  tissue at matched depth", fontsize=9, color=INK, loc="left")
    else:
        ax.text(0.5, 0.5, "WM contacts are the reference\nin this run - see the none run",
                ha="center", va="center", fontsize=8, color=MUTED, transform=ax.transAxes)
        ax.set_title("2  tissue at matched depth", fontsize=9, color=INK, loc="left")
    ax.set_xlabel("depth along shaft (mm)", fontsize=7.5, color=MUTED)

    ax = axes[2]
    if len(top):
        order = top.sort_values("y", ascending=False)
        ax.barh(range(len(order)), order.y, color=[RED if t else BLUE for t in order.temporal])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order.channel, fontsize=6)
        ax.invert_yaxis()
    ax.set_xlabel("mouth − foot at the entry contact (dB)", fontsize=7.5, color=MUTED)
    ax.set_title("3  entry contact per shaft", fontsize=9, color=INK, loc="left")

    ax = axes[3]
    ax.scatter(d.absx, d.y, s=14, c=col, alpha=0.8, lw=0)
    ax.set_xlabel("|x| lateral (mm, fsaverage)", fontsize=7.5, color=MUTED)
    ax.set_title("4  laterality   (picture only)", fontsize=9, color=INK, loc="left")

    for a in axes:
        a.axhline(0, color=GREY, lw=0.7) if a is not axes[2] else a.axvline(0, color=GREY, lw=0.7)
        a.tick_params(labelsize=7, colors=MUTED, length=2)
        for sp in a.spines.values():
            sp.set_color(GREY)
    fig.suptitle(f"{pat}  ·  reref={which}  ·  is the mouth residual shaped like a source "
                 f"outside the skull?", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    keep = ["channel", "shaft", "idx", "temporal", "is_wm", "depth_mm", "absx", "y"]
    d[keep].rename(columns={"y": "mouth_minus_foot"}).to_csv(
        out.replace(".png", ".tsv"), sep="\t", index=False, float_format="%.3f")
    print(f"\n  wrote {out}\n  wrote {out.replace('.png', '.tsv')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="G-05")
    ap.add_argument("--run", default="wm", choices=["wm", "none"])
    a = ap.parse_args()
    run(a.patient, a.run)


if __name__ == "__main__":
    main()
