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
