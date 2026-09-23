#!/usr/bin/env python3
"""
compare_140_hg.py - the third pass: the HG trial rasters, old tree against new.

The cube passes measure the trial-AVERAGED time-frequency map. The HG raster is a
different product of the same run: one row per trial, 70-150 Hz, drawn straight from the
trials that survived the filters. It is blind to the frequency ceiling (the band sits
well inside both), so an HG figure changes only if the TRIALS or the REFERENCE changed -
which makes it the cleanest separator between "the preprocessing moved" and "the data
that went in moved".

Two stages, because reading 13,000 PNG pairs off the share is not free:

  size   every HG figure in both trees, compared by byte size. Free, and a different
         raster always compresses to a different size.
  pixel  a stratified sample plus every contact the cube pass flagged: the mean absolute
         pixel difference after both images are put on the same canvas.

    python compare_140_hg.py                 (size on everything, pixels on a sample)
    python compare_140_hg.py --pixels 400    (bigger pixel sample)

Outputs -> outputs/compare_140/hg_size.tsv, hg_pixels.tsv, summary_hg_patient.tsv
"""
from __future__ import annotations

import argparse
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
CMP = OUTPUTS / "compare_140"
NEW_QC, OLD_QC = OUTPUTS / "04_ersp_LM", OUTPUTS / "04_ersp_LM_old"
CONDS = ("audio", "picture", "reading")
RX = re.compile(r"^(?P<pid>.+?)_(?P<cond>audio|picture|reading)_(?P<ref>[^_]+)_HGtrials_(?P<el>.+)\.png$")


def scan(tree: Path) -> pd.DataFrame:
    rows = []
    if not tree.is_dir():
        return pd.DataFrame(rows)
    for pdir in sorted(p for p in tree.iterdir() if p.is_dir()):
        for cond in CONDS:
            d = pdir / "LM" / "HG" / cond
            if not d.is_dir():
                continue
            with os.scandir(d) as it:
                for e in it:
                    m = RX.match(e.name)
                    if not m:
                        continue
                    rows.append(dict(patient=pdir.name, condition=cond, electrode=m.group("el"),
                                     ref=m.group("ref"), size=e.stat().st_size, path=e.path))
    return pd.DataFrame(rows)


def pixel_diff(args):
    pn, po = args
    try:
        A = plt.imread(pn)
        B = plt.imread(po)
    except Exception as e:
        return np.nan, np.nan, f"read-error: {type(e).__name__}"
    if A.ndim == 3:
        A = A[..., :3].mean(-1)
    if B.ndim == 3:
        B = B[..., :3].mean(-1)
    h, w = min(A.shape[0], B.shape[0]), min(A.shape[1], B.shape[1])
    d = np.abs(A[:h, :w] - B[:h, :w])
    same_shape = A.shape == B.shape
    return float(d.mean()), float((d > 0.02).mean()), ("same canvas" if same_shape
                                                       else f"canvas {B.shape} -> {A.shape}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixels", type=int, default=200, help="pixel-diff sample size")
    a = ap.parse_args()
    CMP.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    N, O = scan(NEW_QC), scan(OLD_QC)
    print(f"[scan] new {len(N)} HG figures · old {len(O)}  ({time.time()-t0:.0f}s)")
    idx = ["patient", "condition", "electrode"]
    J = N.merge(O, on=idx, how="outer", suffixes=("_new", "_old"), indicator=True)
    J["present"] = J._merge.map({"both": "both", "left_only": "only new", "right_only": "only old"})
    J["size_same"] = J.size_new == J.size_old
    J.drop(columns=["_merge"]).to_csv(CMP / "hg_size.tsv", sep="\t", index=False)
    both = J[J.present == "both"]
    print(f"[size] {len(both)} in both · byte-identical size: {int(both.size_same.sum())} "
          f"({100*both.size_same.mean():.1f} %)")

    # pixels: everything the cube pass called most changed, plus a stratified sample
    pick = pd.DataFrame()
    cubes = CMP / "cubes.tsv"
    if cubes.exists():
        C = pd.read_csv(cubes, sep="\t")
        top = (C[C.status == "ok"].sort_values("rmse", ascending=False)
                 .groupby(["patient", "condition"]).head(2)[["patient", "condition", "electrode"]])
        pick = both.merge(top, on=idx)
    rest = both.groupby(["patient", "condition"], group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, a.pixels // max(1, both.patient.nunique() * 3))),
                           random_state=0))
    sel = pd.concat([pick, rest]).drop_duplicates(subset=idx)
    print(f"[pixels] {len(sel)} pairs", flush=True)
    jobs = [(r.path_new, r.path_old) for r in sel.itertuples()]
    res = []
    with ThreadPoolExecutor(10) as ex:
        for k, r in enumerate(ex.map(pixel_diff, jobs)):
            res.append(r)
            if (k + 1) % 100 == 0:
                print(f"    {k+1}/{len(jobs)}", flush=True)
    sel = sel.copy()
    sel["px_mean_abs"], sel["px_frac_changed"], sel["px_note"] = zip(*res)
    sel[idx + ["size_new", "size_old", "size_same", "px_mean_abs", "px_frac_changed", "px_note"]] \
        .to_csv(CMP / "hg_pixels.tsv", sep="\t", index=False)

    g = (J.groupby("patient")
           .agg(n_new=("size_new", "count"), n_old=("size_old", "count"),
                n_both=("present", lambda s: (s == "both").sum()),
                n_only_new=("present", lambda s: (s == "only new").sum()),
                n_only_old=("present", lambda s: (s == "only old").sum()),
                same_size=("size_same", "sum")).reset_index())
    p = sel.groupby("patient").px_mean_abs.median().rename("med_pixel_diff")
    g = g.merge(p, on="patient", how="left")
    g["pct_same_size"] = 100 * g.same_size / g.n_both.replace(0, np.nan)
    g.sort_values("med_pixel_diff", ascending=False).to_csv(
        CMP / "summary_hg_patient.tsv", sep="\t", index=False)
    print(g.sort_values("med_pixel_diff", ascending=False).round(3).to_string(index=False))
    print(f"\nwrote -> {CMP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
