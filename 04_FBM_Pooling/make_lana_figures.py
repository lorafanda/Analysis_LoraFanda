#!/usr/bin/env python3
"""
make_lana_figures.py — LanA / Fedorenko electrode figures on the FIXED recon coords.

Regenerates the language-atlas brain figures after the recon name fix (the aI/al
homoglyph + the four shaft-convention renames), which restored ~118 contacts that
had previously been dropping out of every plot.

Two cohorts, because they answer different questions:

  all      every electrode that has an fsaverage coordinate. "Where did we record?"
  concat   only electrodes carrying ALL THREE conditions, with the grid patients
           (EL044, PAT_3415) and aux/micro channels removed — i.e. exactly the
           sample set the concatenated clustering and stage-04 pooling use.
           "Of the electrodes we actually analyse, which are in the network?"

Three figures per cohort:

  membership_p0.05   in / out of the LanA network at P(language) >= 0.05
  membership_p0.10   ... and at >= 0.10 (the stricter, more selective cut)
  overlap_p          the continuous LanA overlap probability at each electrode

White-matter contacts are KEPT here on purpose. Unlike 252 (which renders cortical
surface plots and so sets KEEP_WM=False), these are scatter plots in fsaverage space
answering "where are the electrodes relative to the atlas" — dropping WM would hide
real recording sites. Pass --no-wm to exclude them.

    python make_lana_figures.py            # write the 6 figures
    python make_lana_figures.py --no-wm    # cortical contacts only
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "04_FBM_Pooling"))
sys.path.insert(0, str(ROOT / "02_FBM_Clustering"))

from functions import lf_atlas_corr as A          # noqa: E402
from functions import lf_brainview as B           # noqa: E402

COORDS = ROOT / "02_FBM_Clustering" / "outputs" / "250_recon" / "fsaverage" / "coords"
ERSP = ROOT / "01_FBM_Analysis" / "outputs" / "04_ersp_LM_RAWONLY"
ATLAS = ROOT / "04_FBM_Pooling" / "federenko_atlas" / "langloc_n806_p_0.05_atlas.nii"
OUT = ROOT / "04_FBM_Pooling" / "outputs" / "pooling" / "atlas_corr" / "fedorenko"

CONDITIONS = ("audio", "picture", "reading")
# Grid / ECoG patients — same rule as lf_concat.DEFAULT_EXCLUDE_PATIENTS.
EXCLUDE_PATIENTS = ("EL044", "PAT_3415")
THRESHOLDS = (0.05, 0.10)

IN_COLOR, OUT_COLOR = "#c0392b", "#b9c0c8"
_RX_ELEC = re.compile(r"_ERSP_(.+?)_TN", re.IGNORECASE)


def norm(s) -> str:
    return "" if s is None else str(s).replace("_", "").replace("-", "").upper()


def load_coords(keep_wm: bool) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(COORDS.glob("*_contacts_fsaverage.csv"))
              if "ALL_PATIENTS" not in f.name.upper()]
    if not frames:
        raise SystemExit(f"no per-patient coords under {COORDS}")
    co = pd.concat(frames, ignore_index=True)
    co["patient_id"] = co["patient"].astype(str)
    co["contact_norm"] = co["name"].map(norm)
    co = co.dropna(subset=["x", "y", "z"])
    if not keep_wm and "is_wm" in co.columns:
        n0 = len(co)
        co = co[co["is_wm"] != 1]
        print(f"  --no-wm: dropped {n0 - len(co)} white-matter contacts")
    co = co.drop_duplicates(["patient_id", "contact_norm"]).reset_index(drop=True)
    return co


def concat_contacts() -> set:
    """(patient, contact) present in ALL THREE conditions, grid patients removed.

    Derived straight from the ERSP tree rather than from a clustering run, so the
    figures already reflect the corrected exclusion list without waiting for 233.
    """
    # Both stages ship a package called `functions`, and 04's shadows 02's on
    # sys.path — load the clustering filters by file path so the aux/micro rules
    # here are literally the same code the clustering track uses.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_lf_dataset", ROOT / "02_FBM_Clustering" / "functions" / "lf_dataset.py")
    _ds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_ds)
    is_non_neural_electrode = _ds.is_non_neural_electrode
    is_micro_electrode = _ds.is_micro_electrode
    per_cond: dict[str, set] = {c: set() for c in CONDITIONS}
    for pdir in sorted(p for p in ERSP.iterdir() if p.is_dir()):
        pid = pdir.name
        if pid in EXCLUDE_PATIENTS:
            continue
        for cond in CONDITIONS:
            d = pdir / "LM" / "ERSP_matrix" / cond
            if not d.is_dir():
                continue
            for f in d.glob("*.npy"):
                m = _RX_ELEC.search(f.stem)
                if not m:
                    continue
                el = m.group(1)
                if is_non_neural_electrode(el) or is_micro_electrode(el, pid):
                    continue
                per_cond[cond].add((pid, norm(el)))
    keep = set.intersection(*(per_cond[c] for c in CONDITIONS))
    print(f"  concat cohort: {len(keep)} electrodes with all 3 conditions "
          f"(excluded {list(EXCLUDE_PATIENTS)})")
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wm", action="store_true", help="drop white-matter contacts")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if not ATLAS.exists():
        raise SystemExit(f"atlas not found: {ATLAS}\n(it is gitignored — re-download it)")

    co = load_coords(keep_wm=not a.no_wm)
    print(f"  coords: {len(co)} contacts, {co['patient_id'].nunique()} patients")

    P = A.sample_atlas_at_contacts(co[["x", "y", "z"]].to_numpy(dtype=float), ATLAS)
    co["P_lana"] = np.nan_to_num(np.asarray(P, dtype=float), nan=0.0)

    keep = concat_contacts()
    co["in_concat"] = [(p, c) in keep for p, c in zip(co["patient_id"], co["contact_norm"])]
    print(f"  of those, {int(co['in_concat'].sum())} are in the concat cohort")

    suffix = "_nowm" if a.no_wm else ""
    rows = []
    for cohort, sub in (("all", co), ("concat", co[co["in_concat"]].reset_index(drop=True))):
        tag = f"{cohort}{suffix}"
        n = len(sub)

        for thr in THRESHOLDS:
            inside = sub["P_lana"].to_numpy() >= thr
            groups = np.where(inside, "in network", "outside")
            n_in = int(inside.sum())
            # Hemisphere asymmetry is the headline claim of the LanA paper, so put the
            # left/right split of the in-network contacts in the title itself.
            hemi = (sub["hemi"].astype(str).str.upper().str[0] if "hemi" in sub.columns
                    else pd.Series(np.where(sub["x"] < 0, "L", "R"), index=sub.index))
            n_lh = int((inside & (hemi == "L")).to_numpy().sum())
            name = f"lana_{cohort}_membership_p{thr:.2f}{suffix}.png"
            B.render_groups(
                sub, groups, colors={"in network": IN_COLOR, "outside": OUT_COLOR},
                order=["outside", "in network"], out_png=OUT / name,
                title=(f"LanA membership · {cohort} electrodes · P(language) >= {thr:g}"
                       f"   —   {n_in}/{n} in network ({100*n_in/max(n,1):.0f}%), "
                       f"{n_lh} left / {n_in-n_lh} right"),
            )
            print(f"  wrote {name}   ({n_in}/{n} in, {n_lh} LH)")
            rows.append(dict(cohort=cohort, figure=name, kind="membership", threshold=thr,
                             n_electrodes=n, n_in=n_in, n_out=n - n_in,
                             pct_in=round(100 * n_in / max(n, 1), 1),
                             n_in_left=n_lh, n_in_right=n_in - n_lh))

        name = f"lana_{cohort}_overlap_p{suffix}.png"
        vals = sub["P_lana"].to_numpy()
        B.render_scalar(
            sub, vals, out_png=OUT / name, cmap="inferno", vmin=0.0,
            vmax=float(np.nanpercentile(vals, 99)) or 1.0,
            label="P(language) — LanA overlap",
            title=(f"LanA overlap probability · {cohort} electrodes · n={n}   "
                   f"(median {np.nanmedian(vals):.3f}, max {np.nanmax(vals):.3f})"),
        )
        print(f"  wrote {name}")
        rows.append(dict(cohort=cohort, figure=name, kind="overlap_p", threshold=np.nan,
                         n_electrodes=n, n_in=np.nan, n_out=np.nan, pct_in=np.nan,
                         n_in_left=np.nan, n_in_right=np.nan))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"lana_figure_summary{suffix}.csv", index=False)
    print()
    print(df.to_string(index=False))
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
