"""Shared by the member_measures steps: where things are, the feature sets, the K they run at, and
the --fset / --k arguments every step accepts.

    --k published   each feature set at its published K (outputs/clustering/bsf_comparison/
                    peak_k.json); only the four feature sets listed there.   -> out/
    --k 7           EVERY feature set at K = 7 (any number works), for a like-for-like comparison,
                    including 15 bands z / s / t.                            -> out_k7/
K-independent tables (white matter, NeuroSynth) live in out/ and are shared by both."""
import argparse
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARE = HERE.parent                                   # 02_FBM_Clustering
OUTC = SHARE / "outputs" / "clustering"
SHARED = HERE / "out"                                 # white matter, NeuroSynth: the same at any K
TAG = {"concat_bands15": "15band", "concat_bands5": "5band", "concat_bands2": "2band", "concat_hfa": "hfa",
       "concat_bands15z": "15zband", "concat_bands15s": "15sband", "concat_bands15t": "15tband"}
LABEL = {"15band": "15 bands", "5band": "5 bands", "2band": "2 bands", "hfa": "HFA",
         "15zband": "15 bands z", "15sband": "15 bands shape", "15tband": "15 bands ternary"}

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--k", default="published")
_ap.add_argument("--fset", default="all")
_ap.add_argument("--ns", default="fdr", choices=["fdr", "nimare"],
                 help="fdr: the NeuroSynth website maps (z > 0 = inside); nimare: 0_nimare_maps.py's "
                      "unthresholded maps (top 10% of each map = inside)")
ARGS, _ = _ap.parse_known_args()
MIN_STUDIES = 200                                     # a map with fewer studies cannot label a cluster
_PUB = json.loads((OUTC / "bsf_comparison" / "peak_k.json").read_text())
if ARGS.k == "published":
    FSETS = {TAG[f]: (f, int(_PUB[f])) for f in TAG if f in _PUB}
    OUT = HERE / "out"
    K_NOTE = "published K"
else:
    FSETS = {TAG[f]: (f, int(ARGS.k)) for f in TAG}
    OUT = HERE / f"out_k{int(ARGS.k)}"
    K_NOTE = f"K = {int(ARGS.k)} for every feature set"
OUT.mkdir(parents=True, exist_ok=True)
# the NeuroSynth tables step 4 writes, and where the figures that use them go
if ARGS.ns == "fdr":
    NS_Z, NS_MAPS, FIG_OUT = SHARED / "neurosynth_z.csv", SHARED / "neurosynth_maps.csv", OUT
else:
    NS_Z, NS_MAPS, FIG_OUT = SHARED / "nimare_z.csv", SHARED / "nimare_maps.csv", OUT / "nimare"
FIG_OUT.mkdir(parents=True, exist_ok=True)


def p2():
    os.environ["PAPER2_PAPER"] = "0"
    if "mm_p2" in sys.modules:
        return sys.modules["mm_p2"]
    m = types.ModuleType("mm_p2")
    m.__file__ = str(SHARE / "00_Paper2_Figures.py")
    sys.modules["mm_p2"] = m
    exec(compile(Path(m.__file__).read_text(encoding="utf-8"), m.__file__, "exec"), m.__dict__)
    return m


def load_module(path, name):
    m = types.ModuleType(name)
    m.__file__ = str(path)
    sys.modules[name] = m
    exec(compile(Path(path).read_text(encoding="utf-8"), m.__file__, "exec"), m.__dict__)
    return m


def tags_from_args():
    """--fset <tag> | all (default all), within the feature sets of the --k mode."""
    if ARGS.fset == "all":
        return list(FSETS)
    if ARGS.fset not in FSETS:
        raise SystemExit(f"--fset {ARGS.fset}: not one of {list(FSETS)} at --k {ARGS.k}")
    return [ARGS.fset]


def fset_dir(tag):
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def cohort_keys():
    """The cohort's electrode keys, from whichever step-1 table exists (every feature set is fitted
    on the same electrodes)."""
    import pandas as pd
    for root in (OUT, HERE / "out", *sorted(HERE.glob("out_k*"))):
        for f in sorted(root.glob("*/measures.csv")):
            return pd.read_csv(f)[["key"]]
    raise SystemExit("run 1_measures.py first")


# ---- shared by steps 13, 15 and 16 (2026-11-07): the two-sided test against the shaft-shift null, and the
# screen / poster / paper versions of a figure

def bh(p):
    """Benjamini-Hochberg q for an array of p (NaN kept as NaN), same shape as p."""
    import numpy as np
    p = np.asarray(p, float)
    q = np.full(p.size, np.nan)
    f = np.isfinite(p.ravel())
    pf = p.ravel()[f]
    if pf.size:
        o = np.argsort(pf)
        qs = np.minimum.accumulate((pf[o] * pf.size / (np.arange(pf.size) + 1))[::-1])[::-1]
        qq = np.empty_like(qs)
        qq[o] = np.minimum(qs, 1)
        q[np.where(f)[0]] = qq
    return q.reshape(p.shape)


def shaft_z(obs, s1, s2, perm, alpha=0.05):
    """obs = the observed shares; s1 / s2 = the sum and the sum of squares of the shares over the `perm` shaft
    shifts (all fractions, any shape). Returns a dict, every share in %: null mean, null SD, z, two-sided p (2 x the
    normal tail beyond |z|), BH q over every finite cell, above / below (q < alpha, by the sign of z), and the corrected
    chance box: mean +/- z_crit x SD, z_crit = the |z| at the BH cut-off, so a share outside the box is exactly a
    significant one (no cell passing: the first BH step, alpha / cells)."""
    import numpy as np
    from scipy.stats import norm
    mean = s1 / perm
    sd = np.sqrt(np.maximum(s2 / perm - mean ** 2, 0) * perm / max(perm - 1, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (obs - mean) / sd, np.nan)
    p = 2 * norm.sf(np.abs(z))
    q = bh(p)
    ok = np.isfinite(q) & (q < alpha)
    p_cut = float(p[ok].max()) if ok.any() else alpha / max(int(np.isfinite(p).sum()), 1)
    zc = float(norm.isf(p_cut / 2))
    return dict(mean=100 * mean, sd=100 * sd, z=z, p=p, q=q, above=ok & (z > 0), below=ok & (z < 0), z_crit=zc,
                lo=np.maximum(100 * (mean - zc * sd), 0.0), hi=100 * (mean + zc * sd))


STYLES = {  # size factors (width, height), text, lines, dpi; the screen version is the figure as drawn
    "screen": dict(w=1.0, h=1.0, text=1.0, line=1.0, dpi=None, suffix=""),
    "poster": dict(w=1.0, h=0.72, text=1.0, line=1.0, dpi=300, suffix="_poster"),     # lower in height
    "paper": dict(w=0.6, h=0.51, text=0.62, line=0.6, dpi=600, suffix="_paper"),      # condensed, smaller text
}


def add_style_arg(ap):
    ap.add_argument("--styles", nargs="+", default=["screen"], choices=list(STYLES),
                    help="which versions to save: screen (the plain file), poster (_poster: lower), paper (_paper: "
                         "condensed, smaller text, 600 dpi)")


def save_styles(fig, path, styles, relayout=None, **savekw):
    """Save `fig` once per style: path (screen), path_poster, path_paper. The figure is resized and its text, lines,
    markers and patch edges scaled, `relayout(fig)` (e.g. tight_layout) re-run when given, then restored."""
    from pathlib import Path as _P
    from matplotlib.text import Text
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    path = _P(path)
    w0, h0 = fig.get_size_inches()
    texts = [(t, t.get_fontsize()) for t in fig.findobj(Text)]
    lines = [(l, l.get_linewidth(), l.get_markersize()) for l in fig.findobj(Line2D)]
    patches = [(pt, pt.get_linewidth()) for pt in fig.findobj(Patch)]
    from matplotlib.legend import Legend
    legs = [(lg, lg._fontsize, list(lg.legend_handles), [t.get_text() for t in lg.texts], lg._loc)
            for lg in fig.findobj(Legend)]
    ticks = []
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lab = axis.get_ticklabels()
            ticks.append((axis, lab[0].get_fontsize() if lab else None))
    written = []
    for st in styles:
        s = STYLES[st]
        fig.set_size_inches(w0 * s["w"], h0 * s["h"])
        for t, f in texts:
            t.set_fontsize(f * s["text"])
        for l, lw, ms in lines:
            l.set_linewidth(lw * s["line"])
            l.set_markersize(ms * s["line"])
        for pt, lw in patches:
            pt.set_linewidth(lw * s["line"])
        for axis, f in ticks:
            if f is not None:
                axis.set_tick_params(labelsize=f * s["text"])
        for lg, f, hs, ls, loc in legs:                                       # handle boxes are sized at creation: rebuild
            lg.prop.set_size(f * s["text"])
            lg._fontsize = f * s["text"]
            lg._init_legend_box(hs, ls)
            lg.set_loc(loc)
            lg.legendPatch.set_mutation_scale(f * s["text"])
        if relayout is not None:
            relayout(fig)
        out = path.with_name(path.stem + s["suffix"] + path.suffix)
        kw = dict(savekw)
        if s["dpi"]:
            kw["dpi"] = s["dpi"]
        fig.savefig(out, **kw)
        written.append(out)
    fig.set_size_inches(w0, h0)
    for t, f in texts:
        t.set_fontsize(f)
    for l, lw, ms in lines:
        l.set_linewidth(lw)
        l.set_markersize(ms)
    for pt, lw in patches:
        pt.set_linewidth(lw)
    for axis, f in ticks:
        if f is not None:
            axis.set_tick_params(labelsize=f)
    for lg, f, hs, ls, loc in legs:
        lg.prop.set_size(f)
        lg._fontsize = f
        lg._init_legend_box(hs, ls)
        lg.set_loc(loc)
        lg.legendPatch.set_mutation_scale(f)
    return written
