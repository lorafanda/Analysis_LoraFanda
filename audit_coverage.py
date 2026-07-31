#!/usr/bin/env python3
"""
audit_coverage.py — where do electrodes DROP OUT of each pipeline?

The question this answers: when a cluster / class / pool is drawn on the brain, are
all of its members actually shown? Every stage silently loses electrodes — a missing
ERSP file, a contact with no fsaverage coordinate, a name that fails the join — and
those losses never surface in the figures.

Reports ONLY problems. A clean run prints almost nothing.

Chain audited
  1  ERSP      per patient x condition — electrodes missing a condition
  2  RECON     ERSP electrodes with NO fsaverage coordinate  (can never be plotted)
  3  CLUSTER   each run: samples -> how many can be drawn; per-cluster shortfall
  4  POOL      role table -> contacts_pool.csv; per-role shortfall
  5  CLASSIFY  cached task matrices -> coordinate coverage

Usage
    python audit_coverage.py                  # full audit
    python audit_coverage.py --stage cluster  # one stage
    python audit_coverage.py --csv out.csv    # also dump findings as a table
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ERSP_DIR = ROOT / "01_FBM_Analysis" / "outputs" / "04_ersp_LM_RAWONLY"
CLUST_DIR = ROOT / "02_FBM_Clustering" / "outputs" / "clustering"
COORDS_DIR = ROOT / "02_FBM_Clustering" / "outputs" / "250_recon" / "fsaverage" / "coords"
POOL_DIR = ROOT / "04_FBM_Pooling" / "outputs" / "pooling"
CLASSIFY_CACHE = ROOT / "03_FBM_Classifying" / "outputs" / "_dataset" / "classification"
CONDITIONS = ("audio", "picture", "reading")
WORKLIST = POOL_DIR / "missing_coords_worklist.csv"

FINDINGS: list[dict] = []
_RX_BERN = re.compile(r"_ERSP_([A-Za-z]+)_([LR])(\d+)_TN", re.IGNORECASE)


def note(stage, patient, item, problem, detail=""):
    FINDINGS.append(dict(stage=stage, patient=patient, item=item, problem=problem, detail=detail))


def norm(s) -> str:
    """'aH_R-1' -> 'AHR1'. Same rule coords / recon / pooling use."""
    return "" if s is None else str(s).replace("_", "").replace("-", "").upper()


# Non-neural channels are SUPPOSED to have no coordinate, so they must not be reported
# as problems. Use the pipeline's own rule where possible so the two stay in sync.
try:
    sys.path.insert(0, str(ROOT / "02_FBM_Clustering"))
    from functions.lf_dataset import is_non_neural_electrode as _is_non_neural  # type: ignore
    from functions.lf_dataset import is_micro_electrode as _is_micro            # type: ignore
except Exception:                                                   # standalone fallback
    def _is_micro(label, patient_id=None):
        if patient_id is not None and not str(patient_id).upper().startswith("PAT"):
            return False
        sh = re.sub(r"\d+$", "", norm(label))
        return bool(sh) and sh.endswith("M")

    def _is_non_neural(label):
        s = norm(label)
        return (not s) or bool(re.fullmatch(r"(AINP\d*|[XE]\d+)", s)) or \
               any(t in s for t in ("PHOTO", "MRK", "MKR", "ECG", "EKG", "EMG", "AUDIO"))


def looks_non_neural(label) -> bool:
    """Pipeline rule, plus the aux patterns it doesn't cover yet (X9+ etc.)."""
    s = norm(label)
    if not s:
        return True
    if _is_non_neural(label) or _is_non_neural(s):
        return True
    return bool(re.fullmatch(r"X\d+", s))       # X1..X99 aux — filter only covers X1-X8


_RX_ELEC = re.compile(r"_ERSP_(.+?)_TN", re.IGNORECASE)


def electrode_from_filename(name: str) -> str:
    """'<pid>_<cond>_WM_ERSP_<electrode>_TN.npy' -> '<electrode>'. Works for both
    cohorts; BERN electrodes look like 'A_L10', GVA like 'GE4'."""
    m = _RX_ELEC.search(Path(str(name).replace("\\", "/")).name)
    return m.group(1) if m else ""


def contact_key(electrode: str, file_path: str = "") -> str:
    """BERN filenames encode the contact as <stem>_<L|R><num>; GVA carry it directly.
    Falls back to the electrode string, and to parsing it out of the filename when
    walking the ERSP tree (where there is no electrode column)."""
    fname = Path(str(file_path).replace("\\", "/")).name if file_path else ""
    m = _RX_BERN.search(fname) if fname else None
    if m:
        return norm(f"{m.group(1)}{m.group(2)}{m.group(3)}")
    if electrode:
        return norm(electrode)
    return norm(electrode_from_filename(fname))


def hdr(txt):
    print(f"\n{'='*74}\n{txt}\n{'='*74}")


def ok(txt):
    print(f"  [ok] {txt}")


# ------------------------------------------------------------------ 1. ERSP
def audit_ersp() -> pd.DataFrame:
    hdr("1 · ERSP — electrodes missing one or more conditions")
    if not ERSP_DIR.exists():
        print(f"  !! ERSP dir not found: {ERSP_DIR}"); return pd.DataFrame()
    rows = []
    for pdir in sorted(p for p in ERSP_DIR.iterdir() if p.is_dir()):
        pid = pdir.name
        per_cond = {}
        for cond in CONDITIONS:
            d = pdir / "LM" / "ERSP_matrix" / cond
            per_cond[cond] = {contact_key("", f.name): f.name
                              for f in d.glob("*.npy")} if d.is_dir() else {}
        if not any(per_cond.values()):
            note("ersp", pid, "-", "no ERSP output at all")
            print(f"  {pid}: NO ERSP OUTPUT")
            continue
        allc = set().union(*[set(v) for v in per_cond.values()])
        for c in sorted(allc):
            missing = [cond for cond in CONDITIONS if c not in per_cond[cond]]
            rows.append(dict(patient_id=pid, contact_norm=c,
                             n_cond=len(CONDITIONS) - len(missing)))
            if missing:
                note("ersp", pid, c, f"missing {len(missing)} condition(s)", ",".join(missing))
        n_part = sum(1 for c in allc if any(c not in per_cond[cond] for cond in CONDITIONS))
        if n_part:
            print(f"  {pid}: {n_part}/{len(allc)} electrodes missing >=1 condition "
                  f"(per-cond counts: " + ", ".join(f"{c}={len(per_cond[c])}" for c in CONDITIONS) + ")")
    inv = pd.DataFrame(rows)
    if inv.empty:
        print("  !! no ERSP electrodes found")
    elif not any(f["stage"] == "ersp" for f in FINDINGS):
        ok(f"all {len(inv)} electrodes have all three conditions")
    return inv


# ------------------------------------------------------------------ 2. coords
def audit_coords(inv: pd.DataFrame) -> set:
    hdr("2 · RECON — ERSP electrodes with NO fsaverage coordinate (cannot be plotted)")
    if not COORDS_DIR.exists():
        print(f"  !! coords dir not found: {COORDS_DIR}"); return set()
    frames = [pd.read_csv(f) for f in sorted(COORDS_DIR.glob("*_contacts_fsaverage.csv"))
              if "ALL_PATIENTS" not in f.name.upper()]
    if not frames:
        print("  !! no per-patient coords CSVs"); return set()
    co = pd.concat(frames, ignore_index=True)
    co["patient_id"] = co["patient"].astype(str)
    co["contact_norm"] = co["name"].map(norm)

    # A coordinate row can exist and still be unusable.
    if {"x", "y", "z"} <= set(co.columns):
        nan_xyz = co[co[["x", "y", "z"]].isna().any(axis=1)]
        if len(nan_xyz):
            print(f"  !! {len(nan_xyz)} coord rows have NaN xyz "
                  f"({sorted(nan_xyz['patient_id'].unique())[:6]})")
            note("coords", "-", f"{len(nan_xyz)} rows", "coord row present but xyz is NaN")
        co = co[~co[["x", "y", "z"]].isna().any(axis=1)]
    dup = co.duplicated(["patient_id", "contact_norm"], keep=False)
    if dup.any():
        d = co[dup]
        print(f"  !! {int(dup.sum())} duplicated (patient, contact) coord rows — joins may fan out "
              f"({sorted(d['patient_id'].unique())[:6]})")
        note("coords", "-", f"{int(dup.sum())} rows", "duplicate contact in coords CSV",
             ";".join(sorted(d['patient_id'].unique())[:12]))

    have = set(zip(co["patient_id"], co["contact_norm"]))
    print(f"  coords available for {len(have)} contacts across {co['patient_id'].nunique()} patients")

    # Patients on one side of the pipeline but not the other.
    if not inv.empty:
        ersp_pats = set(inv["patient_id"]); coord_pats = set(co["patient_id"])
        only_coords = sorted(coord_pats - ersp_pats)
        only_ersp = sorted(ersp_pats - coord_pats)
        if only_ersp:
            print(f"  !! patients with ERSP but NO coords file at all: {only_ersp}")
            note("coords", ",".join(only_ersp), "-", "patient has ERSP but no coords file")
        if only_coords:
            print(f"  (patients with coords but no ERSP yet — not an error: {only_coords})")
    if inv.empty:
        return have
    inv = inv.copy()
    inv["has_xyz"] = list(zip(inv["patient_id"], inv["contact_norm"]))
    inv["has_xyz"] = inv["has_xyz"].map(lambda k: k in have)
    bad = inv[~inv["has_xyz"]].copy()
    bad["aux"] = [looks_non_neural(c) or _is_micro(c, p)
                  for p, c in zip(bad["patient_id"], bad["contact_norm"])]
    real, aux = bad[~bad["aux"]], bad[bad["aux"]]
    if real.empty:
        ok("every real electrode has a coordinate")
    else:
        print("  REAL electrodes with no coordinate — these silently vanish from every brain plot:")
        for pid, g in real.groupby("patient_id"):
            tot = int((inv["patient_id"] == pid).sum())
            shafts = sorted({re.sub(r"\d+$", "", c) for c in g["contact_norm"]})
            print(f"    {pid}: {len(g)}/{tot} missing · shaft(s): {', '.join(shafts[:8])}")
            note("coords", pid, f"{len(g)} contacts", "no fsaverage coordinate",
                 "shafts=" + ",".join(shafts[:12]))
    if not aux.empty:
        print(f"  ({len(aux)} aux / micro contacts also lack coords — expected by design, not a problem)")

    # Per-patient work list for whoever redoes the localisation.
    if not real.empty:
        wl = real[["patient_id", "contact_norm", "n_cond"]].copy()
        wl["shaft"] = wl["contact_norm"].str.replace(r"\d+$", "", regex=True)
        wl = wl.sort_values(["patient_id", "shaft", "contact_norm"])
        wl.to_csv(WORKLIST, index=False)
        print(f"  -> per-contact work list ({len(wl)} contacts): {WORKLIST.name}")
    return have


# ------------------------------------------------------------------ 3. clustering
def audit_clustering(have: set):
    hdr("3 · CLUSTERING — can every clustered sample be drawn on the brain?")
    idx_p = CLUST_DIR / "index.json"
    if not idx_p.exists():
        print(f"  !! no {idx_p}"); return
    runs = json.loads(idx_p.read_text()).get("runs", [])
    print(f"  {len(runs)} runs in index.json")
    for r in runs:
        rd = CLUST_DIR / r.get("path", "")
        lab_p = rd / "labels.csv"
        tag = f"{r.get('method')}/{r.get('feature_set')}/{r.get('run_id')}"
        if not lab_p.exists():
            print(f"  {tag}: !! labels.csv MISSING"); note("cluster", "-", tag, "labels.csv missing"); continue
        lab = pd.read_csv(lab_p)
        ccols = [c for c in lab.columns if c.startswith("cluster_") and not c.endswith("_ranked")]
        if not ccols:
            note("cluster", "-", tag, "no cluster column"); continue
        cc = ccols[0]
        key = [contact_key(e, f) for e, f in
               zip(lab.get("electrode", ""), lab.get("file_path", [""] * len(lab)))]
        lab["_ok"] = [(p, k) in have for p, k in zip(lab["patient_id"].astype(str), key)]
        miss = int((~lab["_ok"]).sum())
        recon = rd / "recon"
        if miss:
            print(f"  {tag}: {miss}/{len(lab)} samples CANNOT be plotted (no coords)")
            note("cluster", "-", tag, f"{miss}/{len(lab)} samples unplottable")
            per = lab[~lab["_ok"]].groupby(cc).size().sort_values(ascending=False)
            for cid, n in per.head(8).items():
                tot = int((lab[cc] == cid).sum())
                print(f"      cluster {cid}: {n}/{tot} missing ({100*n/tot:.0f}% of that cluster)")
        if not recon.is_dir():
            print(f"  {tag}: no recon/ — run 252 to render it")
            note("cluster", "-", tag, "no recon rendered")
        else:
            drawn = sorted(d.name for d in recon.iterdir() if d.is_dir() and d.name.startswith("cluster_"))
            expect = sorted(f"cluster_{int(c):02d}" for c in lab[cc].dropna().unique())
            gone = set(expect) - set(drawn)
            if gone:
                print(f"  {tag}: recon MISSING dirs for {sorted(gone)}")
                note("cluster", "-", tag, "recon missing clusters", ",".join(sorted(gone)))
            # a cluster dir can exist and still hold no rendered views
            empty = [d for d in drawn if not list((recon / d).rglob("*.png"))]
            if empty:
                print(f"  {tag}: recon dirs with NO png: {empty[:6]}")
                note("cluster", "-", tag, "recon dirs empty", ",".join(empty[:12]))
            # cross-check my count against 252's own unmatched log
            um = recon / "UNMATCHED_contacts.csv"
            if um.exists():
                try:
                    n_um = len(pd.read_csv(um))
                    if abs(n_um - miss) > 1:
                        print(f"  {tag}: !! my unplottable count ({miss}) disagrees with "
                              f"252's UNMATCHED log ({n_um}) — one of the joins is wrong")
                        note("cluster", "-", tag, "audit vs 252 unmatched mismatch",
                             f"audit={miss} recon_log={n_um}")
                except Exception:
                    pass
        if not miss and recon.is_dir():
            ok(f"{tag}: all {len(lab)} samples plottable, recon present")


# ------------------------------------------------------------------ 4. pooling
def audit_pooling(have: set):
    hdr("4 · POOLING — do all matched contacts reach contacts_pool.csv?")
    for sub in ("pool_web", "pool_web_gaussian"):
        d = POOL_DIR / sub
        cp = d / "contacts_pool.csv"
        if not cp.exists():
            print(f"  {sub}: no contacts_pool.csv (not exported)"); continue
        pool = pd.read_csv(cp)
        print(f"  {sub}: {len(pool)} contacts exported")
        bad = pool[pool[["x", "y", "z"]].isna().any(axis=1)] if {"x","y","z"} <= set(pool.columns) else pool.iloc[0:0]
        if len(bad):
            print(f"    !! {len(bad)} rows have no xyz — they will not render")
            note("pool", "-", sub, f"{len(bad)} exported rows missing xyz")
        # Role table vs export — MUST compare the same grid, or this is meaningless
        # (the ds and full tables match different numbers of contacts by design).
        grid = None
        pidx = d / "pool_index.json"
        if pidx.exists():
            try:
                grid = json.loads(pidx.read_text()).get("grid")
            except Exception:
                grid = None
        runs = sorted((POOL_DIR / "roles_raw" / grid).glob("runs/*/role_table.parquet")) if grid else []
        if not runs:
            print(f"    (no {grid or '?'}-grid role_table to compare against — skipping export check)")
        else:
            rt = pd.read_parquet(runs[-1])
            if "role" in rt.columns and "role" in pool.columns:
                n_rt = int(rt["role"].astype(str).ne("none").sum())
                n_pool = int(pool["role"].astype(str).ne("none").sum())
                if n_rt - n_pool > 0:
                    print(f"    !! {n_rt - n_pool} role-matched contacts did not reach the export "
                          f"({grid} role table {n_rt} vs pool {n_pool}) — usually missing coords")
                    note("pool", "-", sub, f"{n_rt-n_pool} matched contacts lost before export",
                         f"grid={grid} run={runs[-1].parent.name}")
                else:
                    ok(f"{sub}: all {n_rt} matched contacts ({grid} grid) reached the export")
        if "role" in pool.columns:
            for role, g in pool[pool["role"].astype(str) != "none"].groupby("role"):
                nx = int(g[["x", "y", "z"]].notna().all(axis=1).sum()) if {"x","y","z"} <= set(g.columns) else len(g)
                if nx < len(g):
                    print(f"    role {role}: only {nx}/{len(g)} plottable")
                    note("pool", "-", f"{sub}:{role}", f"{len(g)-nx}/{len(g)} unplottable")
        # role cards the web info panel expects
        ri = d / "role_info.json"
        if ri.exists() and "role" in pool.columns:
            info = json.loads(ri.read_text(encoding="utf-8"))
            for role, meta in info.items():
                n_prim = int((pool["role"].astype(str) == role).sum())
                if n_prim and not meta.get("img"):
                    print(f"    role {role}: {n_prim} contacts but NO card image")
                    note("pool", "-", f"{sub}:{role}", "role has members but no card image")

    # poolv2 designer must cover every exported contact, or its counts silently differ
    cube_man = POOL_DIR / "pool_cube" / "cube_manifest.json"
    cp = POOL_DIR / "pool_web" / "contacts_pool.csv"
    if cube_man.exists() and cp.exists():
        keys = set(json.loads(cube_man.read_text()).get("contacts", []))
        pool = pd.read_csv(cp)
        want = {f"{p}|{c}" for p, c in zip(pool["patient"].astype(str), pool["contact"].astype(str))}
        gone = want - keys
        if gone:
            print(f"  poolv2 cube is MISSING {len(gone)} exported contacts "
                  f"(designer counts will differ from the map)")
            note("pool", "-", "pool_cube", f"{len(gone)} exported contacts absent from cube")
        else:
            ok(f"pool_cube covers all {len(want)} exported contacts")


# ------------------------------------------------------------------ 5. classification
def audit_classification(have: set):
    hdr("5 · CLASSIFICATION — coordinate coverage of the cached task matrices")
    if not CLASSIFY_CACHE.exists():
        print(f"  no cache at {CLASSIFY_CACHE} (run 310)"); return
    metas = sorted(CLASSIFY_CACHE.rglob("meta.parquet"))
    if not metas:
        print("  no meta.parquet found (run 310)"); return
    seen = set()
    for mp in metas:
        task = mp.parent.relative_to(CLASSIFY_CACHE).as_posix()
        key = task.rsplit("/", 1)[0]
        if key in seen:
            continue
        seen.add(key)
        md = pd.read_parquet(mp)
        if "patient_id" not in md.columns:
            continue
        if "contact_norm" in md.columns:            # parcellation caches store the key directly
            k = [norm(c) for c in md["contact_norm"]]
        else:
            el = md.get("electrode", pd.Series([""] * len(md)))
            fp = md.get("file_path", pd.Series([""] * len(md)))
            k = [contact_key(e, f) for e, f in zip(el, fp)]
        okm = [(p, kk) in have for p, kk in zip(md["patient_id"].astype(str), k)]
        miss = len(okm) - sum(okm)
        if miss:
            print(f"  {key}: {miss}/{len(md)} samples have no coords (cannot be mapped)")
            note("classify", "-", key, f"{miss}/{len(md)} samples without coords")
        else:
            ok(f"{key}: all {len(md)} samples have coords")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["ersp", "coords", "cluster", "pool", "classify", "all"], default="all")
    ap.add_argument("--csv", help="write findings to this CSV")
    a = ap.parse_args()

    print(f"AUDIT · {ROOT}")
    inv = audit_ersp() if a.stage in ("all", "ersp", "coords", "cluster", "pool", "classify") else pd.DataFrame()
    have = audit_coords(inv) if a.stage in ("all", "coords", "cluster", "pool", "classify") else set()
    if a.stage in ("all", "cluster"):
        audit_clustering(have)
    if a.stage in ("all", "pool"):
        audit_pooling(have)
    if a.stage in ("all", "classify"):
        audit_classification(have)

    hdr("SUMMARY")
    if not FINDINGS:
        print("  no problems found.")
    else:
        df = pd.DataFrame(FINDINGS)
        for st, g in df.groupby("stage"):
            print(f"  {st:9s} {len(g):5d} finding(s)")
        if a.csv:
            df.to_csv(a.csv, index=False)
            print(f"\n  findings -> {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
