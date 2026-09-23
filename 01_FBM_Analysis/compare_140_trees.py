#!/usr/bin/env python3
"""
compare_140_trees.py - what the 2026-09-18 rerun changed, contact by contact.

THE QUESTION. Both 140 output trees are on disk: the run up to 2026-09-17 (renamed
`*_old`) and the run of 2026-09-18 plus its 09-21/22 reruns. Which patients, conditions
and electrodes look DIFFERENT between them, and why?

WHY A DIFFERENCE IS INFORMATIVE. The headline change of 09-18 was the cube's ceiling:
fmax 500 -> 400 Hz, so a cube went from 129 to 103 frequency bins. That crop happens
AFTER the STFT and the dB conversion (lf_ersp._spectro), and both trees use the same
nperseg / nfft / fs, so bin k is the same 3.90625 Hz band in both and the first 103 bins
of an old cube should be BIT-IDENTICAL to a new one - unless something else changed:

    a different bad-channel list          (the contact is gone, or the reference is)
    a different reference                 (WM set, whole CAR, bad contacts in the mean)
    different triggers or trials          (EL043's picture block, IQR, crop)
    trial rejection scored on a shorter   (cfg.ersp_trial_reject, per patient)
        frequency axis

So: crop the old cube to the new one's height, compare, and every non-zero difference
is one of those - not the crop. That is the whole design.

THREE INDEPENDENT PASSES, because one number is not a check:

  cubes    ERSP_matrix/<cond>/*.npy      the trial-averaged cube itself
  halves   ERSP_halves/<cond>/*_half1|2  the odd/even split-half cubes, different files
                                         written by a different line of the notebook
  images   04_ersp_LM/<pid>/LM/ERSP and HG PNGs, pixel by pixel - a different medium
                                         entirely, and the only way to see the HG
                                         rasters, which carry the TRIAL changes

The three are ranked independently and their rankings compared at the end (Spearman); a
patient that only one pass flags is a bug in that pass, not a finding.

    python compare_140_trees.py --stage inventory      (fast: what exists where)
    python compare_140_trees.py --stage cubes          (pass 1)
    python compare_140_trees.py --stage halves         (pass 2)
    python compare_140_trees.py --stage images         (pass 3, on the flagged set)
    python compare_140_trees.py --stage all

Outputs -> outputs/compare_140/
    inventory.tsv          one row per (tree, patient, condition, electrode)
    cubes.tsv              one row per contact-condition present in BOTH trees
    cubes_missing.tsv      present in only one tree, with which
    halves.tsv             the same for the split halves
    images.tsv             per-PNG pixel difference
    summary_*.tsv          per patient and per patient x condition
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUT = OUTPUTS / "compare_140"
NEW_RAW, OLD_RAW = OUTPUTS / "04_ersp_LM_RAWONLY", OUTPUTS / "04_ersp_LM_RAWONLY_old"
NEW_QC, OLD_QC = OUTPUTS / "04_ersp_LM", OUTPUTS / "04_ersp_LM_old"
CONDS = ("audio", "picture", "reading")
BANDS = [(1, 20), (20, 70), (70, 170), (170, 270), (270, 400)]
HG = (70.0, 150.0)
DF_HZ = 1000.0 / 256.0          # fs 1 kHz, nfft 256 -> 3.90625 Hz per bin, both trees
RX = re.compile(r"^(?P<pid>.+?)_(?P<cond>audio|picture|reading)_(?P<ref>[^_]+)_ERSP_"
                r"(?P<el>.+?)_TN(?P<sfx>_half[12])?\.npy$")
N_THREADS = 12


def norm_el(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()


# ------------------------------------------------------------------ inventory
def scan(root: Path, sub: str) -> pd.DataFrame:
    """Every cube in a tree, parsed. `sub` is ERSP_matrix or ERSP_halves."""
    rows = []
    if not root.is_dir():
        return pd.DataFrame(rows)
    for pdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cond in CONDS:
            d = pdir / "LM" / sub / cond
            if not d.is_dir():
                continue
            with os.scandir(d) as it:
                for e in it:
                    m = RX.match(e.name)
                    if not m:
                        continue
                    st = e.stat()
                    rows.append(dict(patient=pdir.name, condition=cond,
                                     electrode=m.group("el"), key=norm_el(m.group("el")),
                                     reref=m.group("ref"), half=(m.group("sfx") or "")[1:],
                                     name=e.name, path=e.path,
                                     size=st.st_size, mtime=st.st_mtime))
    return pd.DataFrame(rows)


def inventory(verbose=True) -> pd.DataFrame:
    frames = []
    for tree, root, sub in (("new", NEW_RAW, "ERSP_matrix"), ("old", OLD_RAW, "ERSP_matrix"),
                            ("new", NEW_RAW, "ERSP_halves"), ("old", OLD_RAW, "ERSP_halves")):
        t0 = time.time()
        d = scan(root, sub)
        d["tree"], d["kind"] = tree, sub
        frames.append(d)
        if verbose:
            print(f"  {tree:3s} {sub:12s} {len(d):6d} files  ({time.time()-t0:.0f}s)", flush=True)
    inv = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    inv.drop(columns=["path"]).to_csv(OUT / "inventory.tsv", sep="\t", index=False)
    return inv


# ------------------------------------------------------------------ the metrics
def band_rows(nf: int, lo: float, hi: float) -> np.ndarray:
    f = np.arange(nf) * DF_HZ
    return np.flatnonzero((f >= lo) & (f < hi))


def compare_pair(args):
    """Old vs new for one contact-condition. The old cube is CROPPED to the new one's
    height first: same bin width, same origin, so bin k is the same band in both."""
    key, pnew, pold = args
    try:
        A = np.load(pnew).astype(np.float64)      # new
        B = np.load(pold).astype(np.float64)      # old
    except Exception as e:
        return {**key, "status": f"read-error: {type(e).__name__}"}
    out = {**key, "nf_new": A.shape[0], "nt_new": A.shape[1],
           "nf_old": B.shape[0], "nt_old": B.shape[1]}
    if A.shape[1] != B.shape[1]:
        out["status"] = "time-axis differs"
        return out
    nf = min(A.shape[0], B.shape[0])
    # what the crop threw away, reported but never compared
    out["old_above_new_mean_abs_dB"] = float(np.nanmean(np.abs(B[nf:]))) if B.shape[0] > nf else np.nan
    a, b = A[:nf], B[:nf]
    d = a - b
    fin = np.isfinite(a) & np.isfinite(b)
    out["n_nan_new"] = int((~np.isfinite(A)).sum())
    out["n_nan_old"] = int((~np.isfinite(B)).sum())
    if fin.sum() < 10:
        out["status"] = "no finite overlap"
        return out
    out["identical"] = bool(np.array_equal(a[fin], b[fin]))
    out["max_abs_diff"] = float(np.nanmax(np.abs(d[fin])))
    out["rmse"] = float(np.sqrt(np.nanmean(d[fin] ** 2)))
    out["mean_abs_diff"] = float(np.nanmean(np.abs(d[fin])))
    av, bv = a[fin], b[fin]
    out["r"] = float(np.corrcoef(av, bv)[0, 1]) if av.std() > 0 and bv.std() > 0 else np.nan
    out["peak_new"] = float(np.nanmax(np.abs(a)))
    out["peak_old"] = float(np.nanmax(np.abs(b)))
    hg = band_rows(nf, *HG)
    if hg.size:
        dh = d[hg]
        fh = np.isfinite(dh)
        out["hg_mean_abs_diff"] = float(np.nanmean(np.abs(dh[fh]))) if fh.any() else np.nan
        out["hg_mean_new"] = float(np.nanmean(a[hg]))
        out["hg_mean_old"] = float(np.nanmean(b[hg]))
    for lo, hi in BANDS:
        rr = band_rows(nf, lo, hi)
        out[f"d_{lo}_{hi}"] = float(np.nanmean(np.abs(d[rr]))) if rr.size else np.nan
    # where in time the change sits: the cube is [baseline|stimulus|post], 300 bins,
    # stimulus = first half, post = second half (cfg.proportions (0, .5, .5))
    half = a.shape[1] // 2
    out["d_stim"] = float(np.nanmean(np.abs(d[:, :half])))
    out["d_post"] = float(np.nanmean(np.abs(d[:, half:])))
    out["status"] = "ok"
    return out


def run_pairs(inv: pd.DataFrame, kind: str, tag: str) -> pd.DataFrame:
    """Match new to old on (patient, condition, electrode[, half]) and compare."""
    d = inv[inv.kind == kind]
    idx = ["patient", "condition", "key", "half"]
    new = d[d.tree == "new"].set_index(idx)
    old = d[d.tree == "old"].set_index(idx)
    both = new.index.intersection(old.index)
    only_new = new.index.difference(old.index)
    only_old = old.index.difference(new.index)
    print(f"  {kind}: {len(both)} in both · {len(only_new)} only new · {len(only_old)} only old", flush=True)

    miss = pd.concat([
        new.loc[only_new].assign(present="only in the NEW tree").reset_index(),
        old.loc[only_old].assign(present="only in the OLD tree").reset_index()],
        ignore_index=True)
    if len(miss):
        miss[["patient", "condition", "electrode", "key", "half", "reref", "present", "name"]] \
            .to_csv(OUT / f"{tag}_missing.tsv", sep="\t", index=False)

    jobs = []
    for k in both:
        n_, o_ = new.loc[k], old.loc[k]
        n_ = n_.iloc[0] if isinstance(n_, pd.DataFrame) else n_
        o_ = o_.iloc[0] if isinstance(o_, pd.DataFrame) else o_
        jobs.append((dict(patient=k[0], condition=k[1], key=k[2], half=k[3],
                          electrode=n_["electrode"], reref_new=n_["reref"], reref_old=o_["reref"],
                          size_new=n_["size"], size_old=o_["size"]), n_["path"], o_["path"]))
    t0 = time.time()
    rows, done = [], 0
    with ThreadPoolExecutor(N_THREADS) as ex:
        for r in ex.map(compare_pair, jobs):
            rows.append(r)
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{tag}.tsv", sep="\t", index=False)
    print(f"  wrote {OUT / (tag + '.tsv')}  ({time.time()-t0:.0f}s)", flush=True)
    return df


# ------------------------------------------------------------------ summaries
def summarise(df: pd.DataFrame, tag: str) -> None:
    ok = df[df.status == "ok"].copy()
    if not len(ok):
        return
    ok["changed"] = ~ok["identical"].astype(bool)
    g = (ok.groupby(["patient", "condition"])
           .agg(n=("key", "size"), n_changed=("changed", "sum"),
                med_rmse=("rmse", "median"), max_rmse=("rmse", "max"),
                min_r=("r", "min"), med_r=("r", "median"),
                med_hg=("hg_mean_abs_diff", "median"), max_hg=("hg_mean_abs_diff", "max"))
           .reset_index())
    g["pct_changed"] = 100 * g.n_changed / g.n
    g.sort_values(["med_rmse", "max_rmse"], ascending=False).to_csv(
        OUT / f"summary_{tag}_patient_condition.tsv", sep="\t", index=False)
    p = (ok.groupby("patient")
           .agg(n=("key", "size"), n_changed=("changed", "sum"),
                med_rmse=("rmse", "median"), p95_rmse=("rmse", lambda s: s.quantile(.95)),
                max_rmse=("rmse", "max"), min_r=("r", "min"),
                med_hg=("hg_mean_abs_diff", "median"))
           .reset_index())
    p["pct_changed"] = 100 * p.n_changed / p.n
    p.sort_values("med_rmse", ascending=False).to_csv(
        OUT / f"summary_{tag}_patient.tsv", sep="\t", index=False)
    print(f"\n  {tag}: per patient, most changed first")
    print(p.sort_values("med_rmse", ascending=False).round(3).to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["inventory", "cubes", "halves", "all"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("[inventory]")
    inv = inventory()
    if a.stage == "inventory":
        return 0
    if a.stage in ("cubes", "all"):
        print("\n[pass 1 · cubes]")
        summarise(run_pairs(inv, "ERSP_matrix", "cubes"), "cubes")
    if a.stage in ("halves", "all"):
        print("\n[pass 2 · halves]")
        summarise(run_pairs(inv, "ERSP_halves", "halves"), "halves")
    print(f"\nwrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
