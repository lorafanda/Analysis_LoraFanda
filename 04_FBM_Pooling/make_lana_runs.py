#!/usr/bin/env python3
"""
make_lana_runs.py — LanA / Fedorenko partitions as clustering runs, so 252 renders
them with the project's own PyVista glassbrain.

Why runs and not a standalone figure script: every other brain figure in this project
is drawn by 252 on the translucent fsaverage glassbrain, and that renderer needs
PyVista. Writing these as ordinary clustering runs means they go through exactly the
same renderer, colour handling and left/right convention as the cluster maps — no
second visual language, and no chance of the two disagreeing about orientation.

Six runs = 2 cohorts x 3 views.

  cohorts
    lana_all      every electrode with an fsaverage coordinate AND an ERSP, any number
                  of conditions present. "Where did we record?"
    lana_concat   only electrodes carrying ALL THREE conditions, grid patients
                  (EL044, PAT_3415) and aux/micro channels removed — the sample set
                  the concatenated clustering and stage-04 pooling actually use.
                  Derived from the ERSP tree, so it already reflects the corrected
                  exclusion list without waiting for 233 to be re-run.

  views (per cohort)
    p005    binary in/out of the LanA network at P(language) >= 0.05
    p010    ... at >= 0.10
    pbins   the CONTINUOUS overlap probability, cut into 5 bands. 252 renders
            categorical clusters, so a graded map has to be banded rather than a
            smooth colourmap — the bands keep the ordering that a binary cut throws away.

All six use the same atlas (langloc_n806_p_0.05); only the cohort and the cut vary.

    python make_lana_runs.py             # write the runs + register them
    python make_lana_runs.py --dry-run   # report only
Then run 252 — it renders any run lacking a populated recon/ folder.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "04_FBM_Pooling"))
from functions import lf_atlas_corr as A                      # noqa: E402

COORDS = ROOT / "02_FBM_Clustering" / "outputs" / "250_recon" / "fsaverage" / "coords"
ERSP = ROOT / "01_FBM_Analysis" / "outputs" / "04_ersp_LM_RAWONLY"
ATLAS = ROOT / "04_FBM_Pooling" / "federenko_atlas" / "langloc_n806_p_0.05_atlas.nii"
CLUST = ROOT / "02_FBM_Clustering" / "outputs" / "clustering"

CONDITIONS = ("audio", "picture", "reading")
EXCLUDE_PATIENTS = ("EL044", "PAT_3415")
METHOD, METHOD_LABEL = "atlas", "Fedorenko/LanA atlas"
PBINS = [0.0, 0.05, 0.10, 0.20, 0.35, 1.01]
_RX_ELEC = re.compile(r"_ERSP_(.+?)_TN", re.IGNORECASE)
STAMP = dt.datetime.now().strftime("%Y%m%d")


def norm(s) -> str:
    return "" if s is None else str(s).replace("_", "").replace("-", "").upper()


def _clustering_filters():
    """Load 02's aux/micro rules by path — 04 ships a `functions` package too and
    shadows them on sys.path."""
    spec = importlib.util.spec_from_file_location(
        "_lf_ds", ROOT / "02_FBM_Clustering" / "functions" / "lf_dataset.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.is_non_neural_electrode, m.is_micro_electrode


def ersp_index() -> pd.DataFrame:
    """One row per (patient, contact): which conditions exist and the audio file path."""
    is_non_neural, is_micro = _clustering_filters()
    rec: dict[tuple, dict] = {}
    for pdir in sorted(p for p in ERSP.iterdir() if p.is_dir()):
        pid = pdir.name
        for cond in CONDITIONS:
            d = pdir / "LM" / "ERSP_matrix" / cond
            if not d.is_dir():
                continue
            for f in d.glob("*.npy"):
                m = _RX_ELEC.search(f.stem)
                if not m:
                    continue
                el = m.group(1)
                if is_non_neural(el) or is_micro(el, pid):
                    continue
                k = (pid, norm(el))
                r = rec.setdefault(k, {"patient_id": pid, "contact_norm": k[1],
                                       "electrode": el, "conds": set(), "file_path": ""})
                r["conds"].add(cond)
                # 252 needs a file_path; the audio block is the canonical first block,
                # matching how lf_concat stamps file_path on concatenated samples.
                if cond == "audio" or not r["file_path"]:
                    if cond == "audio" or not r["file_path"]:
                        r["file_path"] = str(f)
    rows = [{**v, "n_cond": len(v["conds"])} for v in rec.values()]
    df = pd.DataFrame(rows).drop(columns=["conds"])
    return df


def write_run(feature_set: str, feature_label: str, run_id: str, df: pd.DataFrame,
              cluster: np.ndarray, params: dict, apply: bool) -> dict:
    col = f"cluster_{METHOD}_{feature_set}"
    out = df[["patient_id", "contact_norm", "electrode", "file_path"]].copy()
    out.insert(0, "sample_idx", np.arange(len(out)))
    out["task"] = "LM"
    out["condition"] = "concat"
    out["P_lana"] = df["P_lana"].to_numpy()
    out[col] = np.asarray(cluster, dtype=int)
    out["silhouette"] = np.nan

    sizes = out[col].value_counts().sort_index().to_dict()
    man = {
        "schema_version": 1,
        "method": METHOD, "method_label": METHOD_LABEL,
        "feature_set": feature_set, "feature_set_label": feature_label,
        "run_id": run_id,
        "created_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "params": params,
        "summary": {"n_samples": int(len(out)), "n_clusters": int(out[col].nunique()),
                    "cluster_sizes": {str(k): int(v) for k, v in sizes.items()}},
        "notebook": "make_lana_runs.py — rendered by 252",
        "n_clusters": int(out[col].nunique()),
        "artifacts": {"labels": "labels.csv"},
        "silhouette": None,
    }
    rec = {"method": METHOD, "feature_set": feature_set, "run_id": run_id,
           "path": f"{METHOD}/{feature_set}/runs/{run_id}",
           "n_clusters": man["n_clusters"], "n_samples": int(len(out)),
           "silhouette": None, "created_at": man["created_at"]}
    if apply:
        rd = CLUST / METHOD / feature_set / "runs" / run_id
        rd.mkdir(parents=True, exist_ok=True)
        out.to_csv(rd / "labels.csv", index=False)
        (rd / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"  {feature_set}/{run_id}: n={len(out)} k={man['n_clusters']} sizes={sizes}")
    return rec


def register(recs: list[dict], feature_sets: list[tuple[str, str]], apply: bool) -> None:
    ip = CLUST / "index.json"
    idx = json.loads(ip.read_text(encoding="utf-8"))
    # Entries are {"id","label"} dicts. Coerce first — a bare string here is what broke
    # every clustering run once already.
    def ent(e):
        return {"id": e, "label": e} if isinstance(e, str) else e
    idx["methods"] = [ent(m) for m in idx.get("methods", [])]
    idx["feature_sets"] = [ent(f) for f in idx.get("feature_sets", [])]
    have_m = {m["id"] for m in idx["methods"]}
    if METHOD not in have_m:
        idx["methods"].append({"id": METHOD, "label": METHOD_LABEL})
    have_f = {f["id"] for f in idx["feature_sets"]}
    for fid, flabel in feature_sets:
        if fid not in have_f:
            idx["feature_sets"].append({"id": fid, "label": flabel})
            have_f.add(fid)
    by_path = {r["path"]: r for r in idx.get("runs", [])}
    for r in recs:
        by_path[r["path"]] = r
    idx["runs"] = list(by_path.values())
    if apply:
        ip.write_text(json.dumps(idx, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  index.json: {len(idx['runs'])} runs, "
          f"{len(idx['feature_sets'])} feature sets{'' if apply else '  [dry-run]'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    apply = not a.dry_run
    if not ATLAS.exists():
        raise SystemExit(f"atlas missing (gitignored — re-download): {ATLAS}")

    frames = [pd.read_csv(f) for f in sorted(COORDS.glob("*_contacts_fsaverage.csv"))
              if "ALL_PATIENTS" not in f.name.upper()]
    co = pd.concat(frames, ignore_index=True)
    co["patient_id"] = co["patient"].astype(str)
    co["contact_norm"] = co["name"].map(norm)
    co = co.dropna(subset=["x", "y", "z"]).drop_duplicates(["patient_id", "contact_norm"])

    ei = ersp_index()
    df = ei.merge(co[["patient_id", "contact_norm", "x", "y", "z"]],
                  on=["patient_id", "contact_norm"], how="inner").reset_index(drop=True)
    P = A.sample_atlas_at_contacts(df[["x", "y", "z"]].to_numpy(dtype=float), ATLAS)
    df["P_lana"] = np.nan_to_num(np.asarray(P, dtype=float), nan=0.0)
    print(f"  {len(df)} electrodes with ERSP + coords")

    concat_mask = (df["n_cond"] == len(CONDITIONS)) & ~df["patient_id"].isin(EXCLUDE_PATIENTS)
    print(f"  concat cohort: {int(concat_mask.sum())} "
          f"(all 3 conditions, minus {list(EXCLUDE_PATIENTS)})\n")

    cohorts = [("lana_all", "LanA · all electrodes", df),
               ("lana_concat", "LanA · concatenated cohort", df[concat_mask].reset_index(drop=True))]
    recs, fsets = [], []
    for fid, flabel, sub in cohorts:
        fsets.append((fid, flabel))
        for thr in (0.05, 0.10):
            lab = (sub["P_lana"].to_numpy() >= thr).astype(int)   # 1 = in network
            recs.append(write_run(
                fid, flabel, f'{STAMP}_thr{thr:g}'.replace('.', 'p'), sub, lab,
                {"atlas": ATLAS.name, "threshold": thr, "cohort": fid,
                 "cluster_meaning": {"0": f"outside (P < {thr})", "1": f"in network (P >= {thr})"}},
                apply))
        bins = np.digitize(sub["P_lana"].to_numpy(), PBINS[1:-1], right=False)
        recs.append(write_run(
            fid, flabel, f"{STAMP}_pbins", sub, bins,
            {"atlas": ATLAS.name, "bins": PBINS, "cohort": fid,
             "cluster_meaning": {str(i): f"{PBINS[i]:.2f} <= P < {PBINS[i+1]:.2f}"
                                 for i in range(len(PBINS) - 1)}},
            apply))
    register(recs, fsets, apply)
    print("\n  Next: run 252 — it renders any run whose recon/ folder is missing or empty.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
