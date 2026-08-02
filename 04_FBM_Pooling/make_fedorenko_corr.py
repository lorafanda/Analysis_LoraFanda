#!/usr/bin/env python3
"""
make_fedorenko_corr.py — ERSP x P(language) correlation map on the time-frequency plane.

Lab-notes Fig 9. For every one of the 129 x 900 time-frequency bins, correlate that
bin's dB across contacts against the contact's LanA language-atlas probability, then
replot r back onto the TF plane. Spearman, then Benjamini-Hochberg FDR across all
116,100 bins.

This existed only as ad-hoc notebook cells, so the published figure could not be
regenerated. It is a script now.

    python make_fedorenko_corr.py                 # berlin, clean (the published look)
    python make_fedorenko_corr.py --cmap RdBu_r --no-clean
    python make_fedorenko_corr.py --atlas p_0.01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "04_FBM_Pooling"))
from functions import lf_atlas_corr as A                      # noqa: E402

CACHE = ROOT / "04_FBM_Pooling" / "outputs" / "_dataset" / "pooling" / "_raw_ungated"
COORDS = ROOT / "02_FBM_Clustering" / "outputs" / "250_recon" / "fsaverage" / "coords"
ATLAS_DIR = ROOT / "04_FBM_Pooling" / "federenko_atlas"
OUT = ROOT / "04_FBM_Pooling" / "outputs" / "pooling" / "atlas_corr" / "fedorenko"
CONDITIONS = ("audio", "picture", "reading")


def norm(s) -> str:
    return "" if s is None else str(s).replace("_", "").replace("-", "").upper()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="p_0.05", help="p_0.001 | p_0.01 | p_0.05")
    ap.add_argument("--cmap", default="berlin")
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--q", type=float, default=0.05, help="FDR level")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    atlas = ATLAS_DIR / f"langloc_n806_{a.atlas}_atlas.nii"
    if not atlas.exists():
        raise SystemExit(f"atlas missing (gitignored, re-download): {atlas}")

    X3 = np.load(CACHE / "X_3d.npy", mmap_mode="r")
    meta = pd.read_parquet(CACHE / "df_meta.parquet").reset_index(drop=True)
    meta["contact_norm"] = meta["electrode"].map(norm)
    meta["_row"] = np.arange(len(meta))
    print(f"  cube {X3.shape}, meta {meta.shape}")

    # One CONCATENATED sample per electrode: [audio | picture | reading].
    # Same construction as lf_concat, so the bins line up with every other stage.
    keep, rows = [], []
    for (pid, cn), g in meta.groupby(["patient_id", "contact_norm"], sort=True):
        by = {c: g[g.condition == c] for c in CONDITIONS}
        if any(len(by[c]) == 0 for c in CONDITIONS):
            continue
        keep.append([int(by[c].iloc[0]["_row"]) for c in CONDITIONS])
        rows.append({"patient_id": pid, "contact_norm": cn})
    df = pd.DataFrame(rows)
    print(f"  {len(df)} electrodes with all three conditions")

    co = pd.concat([pd.read_csv(f) for f in sorted(COORDS.glob("*_contacts_fsaverage.csv"))
                    if "ALL_PATIENTS" not in f.name.upper()], ignore_index=True)
    co["patient_id"] = co["patient"].astype(str)
    co["contact_norm"] = co["name"].map(norm)
    co = co.dropna(subset=["x", "y", "z"]).drop_duplicates(["patient_id", "contact_norm"])
    df = df.merge(co[["patient_id", "contact_norm", "x", "y", "z"]],
                  on=["patient_id", "contact_norm"], how="inner")
    keep = [keep[i] for i in df.index] if len(df) != len(keep) else keep
    print(f"  {len(df)} also have an fsaverage coordinate")

    P = A.sample_atlas_at_contacts(df[["x", "y", "z"]].to_numpy(float), atlas)
    P = np.nan_to_num(np.asarray(P, float), nan=0.0)

    n_f, n_t = X3.shape[1], X3.shape[2]
    X2d = np.empty((len(df), n_f * n_t * len(CONDITIONS)), dtype=np.float32)
    for i, rr in enumerate(keep[:len(df)]):
        X2d[i] = np.concatenate([np.asarray(X3[r]) for r in rr], axis=1).ravel()
    print(f"  design matrix {X2d.shape}  ({n_f} freq x {n_t*len(CONDITIONS)} time bins)")

    r, n_used = A.tf_correlation_map(X2d, P, method="spearman")
    p = A.r_to_p(r, n_used)
    sig, p_thr = A.fdr_bh(p, q=a.q)
    r_map = r.reshape(n_f, n_t * len(CONDITIONS))
    sig_map = sig.reshape(r_map.shape)
    print(f"  n={n_used} contacts | {sig.sum()} / {sig.size} bins survive FDR q={a.q} "
          f"({100*sig.mean():.1f}%) | r in [{r.min():+.3f}, {r.max():+.3f}]")

    clean = not a.no_clean
    stem = f"fig9_corr_map_{a.cmap}" + ("" if clean else "_annotated")
    A.plot_tf_corr_map(
        r_map, sig_mask=None if clean else sig_map,
        fmax_hz=500.0, cmap=a.cmap, clean=clean,
        title="" if clean else f"ERSP x P(language) — LanA {a.atlas}, n={n_used}",
        cbar_label="Spearman r  vs  P(language)",
        out_png=OUT / f"{stem}.png", dpi=150)
    print(f"  wrote {OUT / (stem + '.png')}")

    pd.DataFrame([{"atlas": a.atlas, "n_contacts": int(n_used), "cmap": a.cmap,
                   "n_sig_bins": int(sig.sum()), "pct_sig": round(100 * float(sig.mean()), 3),
                   "p_thr": p_thr, "r_min": float(r.min()), "r_max": float(r.max())}]
                 ).to_csv(OUT / f"{stem}_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
