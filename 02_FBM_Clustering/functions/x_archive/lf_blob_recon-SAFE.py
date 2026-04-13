# lf_blob_recon.py
from __future__ import annotations

import re, json, hashlib, shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import imageio.v3 as iio
from nibabel.freesurfer.io import read_geometry, read_annot

import functions.lf_blob_recon_config as C


# -------------------------
# Base dir + IO
# -------------------------

def meta_in_path(algo_name: Optional[str] = None) -> Path:
    algo = (algo_name or C.DEFAULT_ALGO_NAME).strip()
    return C.CLUSTERING_ROOT / algo / C.META_IN_FILENAME

def get_base_dir() -> Path:
    if C.BASE_DIR_OVERRIDE:
        return Path(C.BASE_DIR_OVERRIDE)
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def ensure_dirs(algo_name: Optional[str] = None) -> Dict[str, Path]:
    algo = (algo_name or C.DEFAULT_ALGO_NAME).strip()
    atlas_root = C.ATLAS_INPUTS_ROOT_BASE / algo
    atlas_root.mkdir(parents=True, exist_ok=True)
    
    out = {
        "base": get_base_dir(),
        "meta_in": meta_in_path(algo),
        "atlas_inputs": atlas_root,
        "meta_out": atlas_root / C.META_OUT_NAME,
        "coords_out": atlas_root / C.COORDS_OUT_NAME,
        "qc_out": atlas_root / C.QC_OUT_NAME,
        "render_out": atlas_root / C.RENDER_OUT_DIRNAME,
    }
    out["render_out"].mkdir(parents=True, exist_ok=True)
    return out



def read_table_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lmap = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lmap:
            return lmap[cand.lower()]
    return None


# -------------------------
# Meta collapse to contact-level
# -------------------------
def mode_int(series: pd.Series) -> Optional[int]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return None
    m = s.mode()
    return int(m.iloc[0]) if len(m) else int(s.iloc[0])


def collapse_meta_to_contact_level(df_meta: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    cluster_col = str(cluster_col).strip()
    req = {"patient_id", "electrode", cluster_col}
    miss = req - set(df_meta.columns)
    if miss:
        raise ValueError(f"df_meta missing required columns: {sorted(miss)}")

    df = df_meta.copy()
    if C.FILTER_HIGH_ACTIVITY and C.HIGH_ACTIVITY_COL in df.columns:
        df = df.loc[df[C.HIGH_ACTIVITY_COL].astype(bool)].copy()

    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["electrode"]  = df["electrode"].astype(str).str.strip()

    if C.COLLAPSE_CLUSTER_STRATEGY == "mode":
        cluster_agg = (cluster_col, mode_int)
    elif C.COLLAPSE_CLUSTER_STRATEGY == "first":
        cluster_agg = (cluster_col, lambda s: int(pd.to_numeric(s, errors="coerce").dropna().iloc[0]))
    else:
        raise ValueError("COLLAPSE_CLUSTER_STRATEGY must be 'mode' or 'first'")

    agg: Dict[str, tuple] = {"cluster": cluster_agg}
    ccol = first_present(df, C.CONDITION_COL_CANDIDATES)
    if ccol:
        agg["condition"] = (ccol, lambda s: sorted(set(map(str, s)))[0])

    out = df.groupby(["patient_id", "electrode"], as_index=False).agg(**agg)
    out = out.dropna(subset=["cluster"]).copy()
    out["cluster"] = out["cluster"].astype(int)
    out["condition"] = out.get("condition", "NA").astype(str).fillna("NA")
    return out


# -------------------------
# Patient dir resolution
# -------------------------
def patient_freesurfer_dir(pid: str) -> Path:
    pid = str(pid).strip()
    if pid.upper().startswith("PAT_"):
        return C.ROOT_PAT / pid / "anatomy" / "prep" / "freesurfer"
    if pid.startswith("MicroEPI"):
        return C.ROOT_MICRO / pid / "anatomy" / "prep"
    if re.match(r"^(EL|el)\d+$", pid):
        return C.ROOT_BERN_FS / pid.lower()
    raise ValueError(f"Unrecognized patient id: {pid}")


def paper1_contacts_csv(pid: str) -> Path:
    return C.PAPER1_RECONS_ROOT / pid / C.BLOCK_NAME / C.CSV_PRODUCT / f"{pid}{C.CONTACTS_CSV_SUFFIX}"


# -------------------------
# EL Lookup search + parsing
# -------------------------
def find_el_lookup_files(pid: str) -> Optional[Path]:
    pid_l = pid.lower()
    bases = [C.BERN_EL_PROJECT_ROOT / pid_l, C.BERN_EL_PROJECT_ROOT]
    hits = []
    for b in bases:
        if not b.exists():
            continue
        hits += list(b.rglob("Lookup*.xlsx"))
        hits += list(b.rglob("*lookup*.xlsx"))
    # best match containing pid
    hits = [h for h in hits if pid_l in str(h).lower()] + [h for h in hits if pid_l not in str(h).lower()]
    return hits[0] if hits else None


def load_el_contacts_from_lookup(pid: str, xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name=C.LOOKUP_SHEET)
    df.columns = [c.strip() for c in df.columns]

    if C.LOOKUP_TYPE_COL in df.columns:
        types = df[C.LOOKUP_TYPE_COL].astype(str).str.lower()
        df = df.loc[types.isin(C.LOOKUP_KEEP_TYPES)].copy()

    has_native = all(c in df.columns for c in C.LOOKUP_NATIVE_COLS)
    has_mni = all(c in df.columns for c in C.LOOKUP_MNI_COLS)

    if has_native and df[C.LOOKUP_NATIVE_COLS[0]].notna().any():
        xcol,ycol,zcol = C.LOOKUP_NATIVE_COLS
        space = "scannerRAS"
    elif C.ALLOW_EL_MNI_IF_NATIVE_MISSING and has_mni and df[C.LOOKUP_MNI_COLS[0]].notna().any():
        xcol,ycol,zcol = C.LOOKUP_MNI_COLS
        space = "MNI_like"
    else:
        raise FileNotFoundError(f"{pid}: Lookup.xlsx has no usable native coords: {xlsx}")

    label_col = None
    for cand in C.LOOKUP_LABEL_PREF:
        if cand in df.columns and df[cand].notna().any():
            label_col = cand
            break
    if not label_col:
        raise ValueError(f"{pid}: Lookup.xlsx missing label columns {C.LOOKUP_LABEL_PREF}: {xlsx}")

    out = pd.DataFrame({
        "patient_id": pid,
        "electrode": df[label_col].astype(str).str.strip(),
        "x": pd.to_numeric(df[xcol], errors="coerce"),
        "y": pd.to_numeric(df[ycol], errors="coerce"),
        "z": pd.to_numeric(df[zcol], errors="coerce"),
        "isSubdural": 0,
        "source_space": space,
        "source_file": str(xlsx),
    }).dropna(subset=["x","y","z","electrode"])
    out = out.loc[out["electrode"].astype(str).str.len() > 0].copy()
    return out


def load_patient_contacts(pid: str) -> pd.DataFrame:
    pid = str(pid).strip()

    if pid in C.CONTACTS_PATH_OVERRIDES:
        p = Path(C.CONTACTS_PATH_OVERRIDES[pid])
        if not p.exists():
            raise FileNotFoundError(f"{pid}: override contacts path missing: {p}")
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        if "name" in df.columns and "electrode" not in df.columns:
            df = df.rename(columns={"name":"electrode"})
        df["patient_id"] = pid
        df["source_space"] = df.get("source_space", "tkrRAS")
        return df

    p_csv = paper1_contacts_csv(pid)
    if p_csv.exists():
        df = pd.read_csv(p_csv)
        df.columns = [c.strip() for c in df.columns]
        if "name" in df.columns and "electrode" not in df.columns:
            df = df.rename(columns={"name":"electrode"})
        df["patient_id"] = pid
        df["source_space"] = "tkrRAS"
        for c in ["x","y","z"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["electrode"] = df["electrode"].astype(str).str.strip()
        return df.dropna(subset=["x","y","z","electrode"]).copy()

    if re.match(r"^(EL|el)\d+$", pid):
        xlsx = find_el_lookup_files(pid)
        if xlsx:
            return load_el_contacts_from_lookup(pid, xlsx)
        raise FileNotFoundError(
            f"{pid}: No EL contacts source found.\n"
            f"  Tried CSV: {p_csv}\n"
            f"  Searched Lookup.xlsx under: {C.BERN_EL_PROJECT_ROOT}"
        )

    raise FileNotFoundError(f"{pid}: No contacts source found. Tried: {p_csv}")


# -------------------------
# Talairach + mgz helpers
# -------------------------
def load_mgz_matrices(subj_dir: Path) -> Tuple[np.ndarray, np.ndarray, Path]:
    for p in [subj_dir/"mri/brainmask.mgz", subj_dir/"mri/T1.mgz", subj_dir/"mri/orig.mgz"]:
        if p.exists():
            img = nib.load(str(p))
            hdr = img.header
            return np.array(hdr.get_vox2ras(), float), np.array(hdr.get_vox2ras_tkr(), float), p
    raise FileNotFoundError(f"No mgz found in {subj_dir}/mri (brainmask/T1/orig)")


def parse_talairach_xfm(xfm_path: Path) -> np.ndarray:
    if not xfm_path.exists():
        raise FileNotFoundError(f"Missing talairach.xfm: {xfm_path}")
    txt = xfm_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Linear_Transform\s*=\s*([\s\S]*?);", txt)
    if not m:
        raise ValueError(f"Linear_Transform block not found in: {xfm_path}")

    rows = []
    for line in m.group(1).strip().splitlines():
        parts = [p for p in line.replace("\t"," ").split(" ") if p]
        if len(parts) >= 4:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
    if len(rows) != 3:
        raise ValueError(f"Expected 3 rows in talairach.xfm, got {len(rows)}: {xfm_path}")

    aff = np.eye(4, dtype=float)
    aff[:3, :4] = np.array(rows, dtype=float)
    return aff


def apply_affine(points_xyz: np.ndarray, aff_4x4: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, float)
    hom = np.c_[pts, np.ones((pts.shape[0], 1), float)]
    out = (aff_4x4 @ hom.T).T
    return out[:, :3]


def scanner_to_tkr(points_scanner: np.ndarray, vox2ras: np.ndarray, vox2ras_tkr: np.ndarray) -> np.ndarray:
    pts_vox = apply_affine(points_scanner, np.linalg.inv(vox2ras))
    return apply_affine(pts_vox, vox2ras_tkr)


def tkr_to_scanner(points_tkr: np.ndarray, vox2ras: np.ndarray, vox2ras_tkr: np.ndarray) -> np.ndarray:
    pts_vox = apply_affine(points_tkr, np.linalg.inv(vox2ras_tkr))
    return apply_affine(pts_vox, vox2ras)


def subject_tkr_to_fsaverage_tkr(pid: str, points_tkr_subj: np.ndarray) -> Tuple[np.ndarray, Dict[str,str]]:
    subj_dir = patient_freesurfer_dir(pid)
    vox2ras_subj, vox2ras_tkr_subj, subj_mgz = load_mgz_matrices(subj_dir)
    tal_subj = parse_talairach_xfm(subj_dir/"mri/transforms/talairach.xfm")

    vox2ras_fs, vox2ras_tkr_fs, fs_mgz = load_mgz_matrices(C.FSAVERAGE_DIR)
    tal_fs = parse_talairach_xfm(C.FSAVERAGE_DIR/"mri/transforms/talairach.xfm")
    inv_tal_fs = np.linalg.inv(tal_fs)

    pts_scanner_subj = tkr_to_scanner(points_tkr_subj, vox2ras_subj, vox2ras_tkr_subj)
    pts_mni = apply_affine(pts_scanner_subj, tal_subj)
    pts_scanner_fs = apply_affine(pts_mni, inv_tal_fs)
    pts_tkr_fs = scanner_to_tkr(pts_scanner_fs, vox2ras_fs, vox2ras_tkr_fs)

    prov = {
        "pid": pid, "subj_dir": str(subj_dir),
        "subj_mgz_used": str(subj_mgz),
        "subj_tal": str(subj_dir/"mri/transforms/talairach.xfm"),
        "fs_mgz_used": str(fs_mgz),
        "fs_tal": str(C.FSAVERAGE_DIR/"mri/transforms/talairach.xfm"),
        "input_space": "subj_tkrRAS", "space_out": "fsaverage_tkrRAS_via_MNI305_affine",
    }
    return pts_tkr_fs, prov


def subject_scanner_to_fsaverage_tkr(pid: str, points_scanner_subj: np.ndarray) -> Tuple[np.ndarray, Dict[str,str]]:
    subj_dir = patient_freesurfer_dir(pid)
    tal_subj = parse_talairach_xfm(subj_dir/"mri/transforms/talairach.xfm")

    vox2ras_fs, vox2ras_tkr_fs, fs_mgz = load_mgz_matrices(C.FSAVERAGE_DIR)
    tal_fs = parse_talairach_xfm(C.FSAVERAGE_DIR/"mri/transforms/talairach.xfm")
    inv_tal_fs = np.linalg.inv(tal_fs)

    pts_mni = apply_affine(points_scanner_subj, tal_subj)
    pts_scanner_fs = apply_affine(pts_mni, inv_tal_fs)
    pts_tkr_fs = scanner_to_tkr(pts_scanner_fs, vox2ras_fs, vox2ras_tkr_fs)

    prov = {
        "pid": pid, "subj_dir": str(subj_dir),
        "subj_tal": str(subj_dir/"mri/transforms/talairach.xfm"),
        "fs_mgz_used": str(fs_mgz),
        "fs_tal": str(C.FSAVERAGE_DIR/"mri/transforms/talairach.xfm"),
        "input_space": "subj_scannerRAS", "space_out": "fsaverage_tkrRAS_via_MNI305_affine",
    }
    return pts_tkr_fs, prov


# -------------------------
# Cell 1: build atlas inputs
# -------------------------
def build_atlas_inputs(algo_name: Optional[str] = None, cluster_col: Optional[str] = None) -> Dict[str, Path]:
    P = ensure_dirs(algo_name)

    if not P["meta_in"].exists():
        raise FileNotFoundError(f"Missing meta file: {P['meta_in']}")

    use_cluster_col = (cluster_col or C.DEFAULT_CLUSTER_COL_IN_META).strip()

    df_meta_raw = read_table_auto(P["meta_in"])
    df_contact = collapse_meta_to_contact_level(df_meta_raw, cluster_col=use_cluster_col)
    df_contact.to_csv(P["meta_out"], sep="\t", index=False)

    patient_ids = sorted(df_contact["patient_id"].astype(str).str.strip().unique().tolist())
    qc_rows, all_rows = [], []

    for pid in patient_ids:
        try:
            subj_dir = patient_freesurfer_dir(pid)
            if not (subj_dir/"mri/transforms/talairach.xfm").exists():
                raise FileNotFoundError(f"Missing talairach.xfm: {subj_dir/'mri/transforms/talairach.xfm'}")

            dfc = load_patient_contacts(pid)
            pts = dfc[["x","y","z"]].to_numpy(float)
            src = str(dfc.get("source_space","tkrRAS").iloc[0]) if len(dfc) else "tkrRAS"

            if src.lower() == "tkrRAS".lower():
                pts_fs, prov = subject_tkr_to_fsaverage_tkr(pid, pts)
            elif src.lower() in ["scannerras","subj_scannerras","native","native_scannerras"]:
                pts_fs, prov = subject_scanner_to_fsaverage_tkr(pid, pts)
            elif src.lower() == "mni_like":
                pts_fs, prov = subject_scanner_to_fsaverage_tkr(pid, pts)
            else:
                raise ValueError(f"Unknown source_space='{src}' for {pid}")

            out = dfc.copy()
            out["x"], out["y"], out["z"] = pts_fs[:,0], pts_fs[:,1], pts_fs[:,2]
            out["coord_space"] = prov["space_out"]
            all_rows.append(out)

            qc_rows.append({"patient_id":pid,"n_contacts":int(len(out)),"status":"OK",
                           "subj_dir":prov["subj_dir"],"input_space":prov["input_space"],
                           "subj_tal":prov.get("subj_tal",""),"fs_tal":prov["fs_tal"]})
            print(f"[OK] {pid}: transformed {len(out)} contacts -> fsaverage tkrRAS (input={prov['input_space']})")

        except Exception as e:
            qc_rows.append({"patient_id":pid,"n_contacts":0,"status":f"ERROR: {e}",
                            "subj_dir":str(patient_freesurfer_dir(pid)) if re.match(r"^(PAT_|MicroEPI|EL|el)", pid) else ""})
            print(f"[ERROR] {pid}: {e}")

    if not all_rows:
        raise RuntimeError("No patients transformed successfully; check QC output.")

    df_coords = pd.concat(all_rows, ignore_index=True)
    df_coords = df_coords.rename(columns={"name":"electrode"}) if "name" in df_coords.columns and "electrode" not in df_coords.columns else df_coords
    df_coords["patient_id"] = df_coords["patient_id"].astype(str).str.strip()
    df_coords["electrode"]  = df_coords["electrode"].astype(str).str.strip()
    df_coords.to_csv(P["coords_out"], sep="\t", index=False)

    pd.DataFrame(qc_rows).to_csv(P["qc_out"], sep="\t", index=False)
    print(f"[WROTE] {P['meta_out']}  (rows={len(df_contact)})")
    print(f"[WROTE] {P['coords_out']} (rows={len(df_coords)})")
    print(f"[WROTE] {P['qc_out']}")
    return P


# -------------------------
# Rendering: color maps (patients/conditions) per your CSS4 lists
# -------------------------
def cohort_of(pid: str) -> str:
    pid = str(pid).strip()
    if pid.upper().startswith("EL"): return "EL"
    if pid.upper().startswith("PAT_"): return "PAT"
    if pid.startswith("MicroEPI"): return "MICRO"
    return "OTHER"


def build_patient_colors_css4(patients: List[str]) -> Dict[str, Tuple[float,float,float,float]]:
    patients = sorted(set(map(str, patients)))
    pools = {"EL": C.EL_COLOR_NAMES, "PAT": C.PAT_COLOR_NAMES, "MICRO": C.MICRO_COLOR_NAMES}
    out: Dict[str, Tuple[float,float,float,float]] = {}

    for cohort, names in pools.items():
        ps = [p for p in patients if cohort_of(p) == cohort]
        if len(ps) > len(names):
            raise ValueError(f"Not enough {cohort} colors: need {len(ps)}, have {len(names)}")
        for p, cname in zip(ps, names):
            if cname not in mcolors.CSS4_COLORS:
                raise ValueError(f"CSS4 color not found: {cname}")
            out[p] = mcolors.to_rgba(cname)

    other = [p for p in patients if cohort_of(p) == "OTHER"]
    if other:
        raise ValueError(f"Unclassified patients (add palette rule): {other}")

    # ensure no duplicates
    seen = {}
    for p, rgba in out.items():
        key = tuple(round(x, 6) for x in rgba)
        if key in seen:
            raise ValueError(f"Duplicate patient color for {p} and {seen[key]} -> {rgba}")
        seen[key] = p

    return out


def build_condition_colors_css4(conditions: List[str]) -> Dict[str, Tuple[float,float,float,float]]:
    conds = sorted(set(map(str, conditions)))
    if len(conds) > len(C.CONDITION_COLOR_NAMES):
        raise ValueError(f"Not enough condition colors: need {len(conds)}, have {len(C.CONDITION_COLOR_NAMES)}")
    out = {}
    for c, cname in zip(conds, C.CONDITION_COLOR_NAMES):
        if cname not in mcolors.CSS4_COLORS:
            raise ValueError(f"CSS4 color not found: {cname}")
        out[c] = mcolors.to_rgba(cname)
    return out


def build_cluster_color_map(clusters: List[int]) -> Dict[int, Tuple[float,float,float,float]]:
    # vibrant evenly spaced HSV
    import colorsys
    cls = sorted(set(int(x) for x in clusters))
    n = max(1, len(cls))
    out = {}
    for i, cl in enumerate(cls):
        r,g,b = colorsys.hsv_to_rgb((i/n) % 1.0, 0.95, 0.98)
        out[cl] = (float(r), float(g), float(b), 1.0)
    return out


# -------------------------
# Rendering primitives
# -------------------------
def normalize_condition_column(df_meta: pd.DataFrame) -> pd.DataFrame:
    df = df_meta.copy()
    ccol = first_present(df, C.CONDITION_COL_CANDIDATES)
    if not ccol:
        df["condition"] = "NA"
        return df
    if ccol != "condition":
        df = df.rename(columns={ccol: "condition"})
    s = df["condition"].astype(str)
    df["condition"] = s.apply(lambda x: x.split(",")[0].strip() if "," in x else x.strip()).replace({"":"NA","nan":"NA","None":"NA"})
    return df


def merge_coords_and_meta(df_meta: pd.DataFrame, df_coords: pd.DataFrame) -> pd.DataFrame:
    df_meta = normalize_condition_column(df_meta)

    for df in (df_meta, df_coords):
        df["patient_id"] = df["patient_id"].astype(str).str.strip()
        df["electrode"]  = df["electrode"].astype(str).str.strip()

    df_meta["cluster"] = pd.to_numeric(df_meta["cluster"], errors="coerce")
    for c in ["x","y","z"]:
        df_coords[c] = pd.to_numeric(df_coords[c], errors="coerce")

    merged = df_coords.merge(df_meta[["patient_id","electrode","cluster","condition"]],
                             how="left", on=["patient_id","electrode"], validate="m:1")
    merged["matched"] = merged["cluster"].notna()

    print("\n[QC] Merge summary")
    print(f"  - coords rows:            {len(df_coords)}")
    print(f"  - missing cluster labels: {int((~merged['matched']).sum())}")
    print(f"  - missing any x/y/z:      {int(merged[['x','y','z']].isna().any(axis=1).sum())}")

    merged = merged.dropna(subset=["x","y","z","cluster"]).copy()
    merged["cluster"] = merged["cluster"].astype(int)
    merged["condition"] = merged["condition"].astype(str).fillna("NA")

    subd = first_present(merged, C.SUBDURAL_COL_CANDIDATES)
    if subd:
        merged[subd] = pd.to_numeric(merged[subd], errors="coerce")
        before = len(merged)
        keep = np.zeros(len(merged), dtype=bool)
        if C.INCLUDE_DEPTH: keep |= (merged[subd] == 0)
        if C.INCLUDE_SUBDURAL: keep |= (merged[subd] == 1)
        merged = merged.loc[keep].copy()
        print(f"  - kept {len(merged)}/{before} using '{subd}' with INCLUDE_DEPTH={C.INCLUDE_DEPTH}, INCLUDE_SUBDURAL={C.INCLUDE_SUBDURAL}")
    return merged


def load_fsaverage_meshes() -> Tuple[pv.PolyData, pv.PolyData]:
    lh_path = C.FSAVERAGE_DIR / "surf" / "lh.pial"
    rh_path = C.FSAVERAGE_DIR / "surf" / "rh.pial"
    if not lh_path.exists() or not rh_path.exists():
        raise FileNotFoundError(f"Missing fsaverage pial: {lh_path} / {rh_path}")

    lh_v, lh_f = read_geometry(str(lh_path))
    rh_v, rh_f = read_geometry(str(rh_path))

    def to_pv(v, f):
        faces = np.hstack([np.full((f.shape[0],1),3,dtype=np.int64), f]).ravel()
        m = pv.PolyData(v, faces)
        m.compute_normals(inplace=True)
        return m

    return to_pv(lh_v, lh_f), to_pv(rh_v, rh_f)


def load_aparc_rgba() -> Tuple[np.ndarray, np.ndarray]:
    lh_annot = C.FSAVERAGE_DIR / "label" / "lh.aparc.annot"
    rh_annot = C.FSAVERAGE_DIR / "label" / "rh.aparc.annot"
    if not lh_annot.exists() or not rh_annot.exists():
        raise FileNotFoundError(f"Missing aparc annot: {lh_annot} / {rh_annot}")

    def annot_to_rgba(p: Path) -> np.ndarray:
        labels, ctab, _ = read_annot(str(p))
        label_ids = ctab[:, -1]
        rgb = ctab[:, :3].astype(np.uint8)
        rgba = np.c_[rgb, np.full((rgb.shape[0],1),255,dtype=np.uint8)]
        lut = {int(lid): rgba[i] for i, lid in enumerate(label_ids)}
        out = np.zeros((labels.shape[0], 4), dtype=np.uint8)
        default = np.array([200,200,200,255], dtype=np.uint8)
        for i, lid in enumerate(labels):
            out[i] = lut.get(int(lid), default)
        return out

    return annot_to_rgba(lh_annot), annot_to_rgba(rh_annot)


def compute_cameras(bounds):
    xmin,xmax,ymin,ymax,zmin,zmax = bounds
    cx,cy,cz = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2
    size = max(xmax-xmin, ymax-ymin, zmax-zmin)
    d = 2.4 * size
    focal, upz = (cx,cy,cz), (0,0,1)
    return {
        "left":((cx-d,cy,cz),focal,upz),
        "right":((cx+d,cy,cz),focal,upz),
        "frontal":((cx,cy+d,cz),focal,upz),
        "posterior":((cx,cy-d,cz),focal,upz),
        "dorsal":((cx,cy,cz+d),focal,(0,1,0)),
        "ventral":((cx,cy,cz-d),focal,(0,1,0)),
    }


def add_brain_mesh(pl: pv.Plotter, mesh: pv.PolyData, rgba: Optional[np.ndarray], opacity: float):
    if rgba is None:
        pl.add_mesh(mesh, color=C.BRAIN_COLOR, opacity=opacity, smooth_shading=True,
                    specular=C.BRAIN_SPECULAR, specular_power=C.BRAIN_SPECULAR_POWER,
                    ambient=C.BRAIN_AMBIENT, diffuse=C.BRAIN_DIFFUSE)
    else:
        m = mesh.copy(deep=True)
        m.point_data["rgba"] = rgba
        pl.add_mesh(m, scalars="rgba", rgba=True, opacity=opacity, smooth_shading=True,
                    specular=C.BRAIN_SPECULAR, specular_power=C.BRAIN_SPECULAR_POWER,
                    ambient=C.BRAIN_AMBIENT, diffuse=C.BRAIN_DIFFUSE)


def add_electrodes(pl: pv.Plotter, df_plot: pd.DataFrame, color_by: str,
                  cluster_colors, patient_colors, cond_colors):
    subd = first_present(df_plot, C.SUBDURAL_COL_CANDIDATES)

    for _, r in df_plot.iterrows():
        x,y,z = float(r["x"]), float(r["y"]), float(r["z"])
        if color_by == "cluster":
            rgb = cluster_colors[int(r["cluster"])][:3]
        elif color_by == "patient":
            rgb = patient_colors[str(r["patient_id"])][:3]
        elif color_by == "condition":
            rgb = cond_colors[str(r["condition"])][:3]
        else:
            raise ValueError("color_by must be cluster/patient/condition")

        rad = C.DEPTH_RADIUS
        if subd:
            try:
                if int(r[subd]) == 1: rad = C.SUBDURAL_RADIUS
            except Exception:
                pass

        sph = pv.Sphere(radius=rad, center=(x,y,z), theta_resolution=18, phi_resolution=18)
        pl.add_mesh(sph, color=rgb, opacity=C.ELECTRODE_OPACITY)


def render_views(lh, rh, df_plot, out_dir: Path, tag: str,
                 lh_rgba, rh_rgba, brain_opacity: float, color_by: str,
                 cluster_colors, patient_colors, cond_colors):
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds = (
        min(lh.bounds[0], rh.bounds[0]), max(lh.bounds[1], rh.bounds[1]),
        min(lh.bounds[2], rh.bounds[2]), max(lh.bounds[3], rh.bounds[3]),
        min(lh.bounds[4], rh.bounds[4]), max(lh.bounds[5], rh.bounds[5]),
    )
    cams = compute_cameras(bounds)

    for view in C.VIEWS_TO_SAVE:
        pl = pv.Plotter(off_screen=True, window_size=C.WINDOW_SIZE)
        add_brain_mesh(pl, lh, lh_rgba, brain_opacity)
        add_brain_mesh(pl, rh, rh_rgba, brain_opacity)
        add_electrodes(pl, df_plot, color_by, cluster_colors, patient_colors, cond_colors)
        pl.camera_position = cams[view]
        pl.reset_camera_clipping_range()
        out_png = out_dir / f"{tag}_{view}.png"
        pl.screenshot(str(out_png), transparent_background=C.TRANSPARENT_BG, scale=C.SS_SCALE)
        pl.close()
        print(f"[WROTE] {out_png}")


def legend_png(color_map: Dict[str, Tuple[float,float,float,float]], title: str, out_png: Path, sort_key=None):
    items = list(color_map.items())
    items = sorted(items, key=lambda kv: sort_key(kv[0]) if sort_key else str(kv[0]))
    labels = [str(k) for k,_ in items]
    colors = [v for _,v in items]
    handles = [mpatches.Patch(color=c[:3], label=lab) for lab, c in zip(labels, colors)]

    n = len(handles)
    ncols = 4 if n > 30 else 3 if n > 18 else 2 if n > 10 else 1
    fig_h = 1.2 + 0.32 * ((n + ncols - 1) // ncols)
    fig_w = 11 if ncols >= 3 else 8.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.legend(handles=handles, ncol=ncols, loc="center", frameon=False,
               fontsize=10, title=title, title_fontsize=11)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# -------------------------
# Patient coverage mosaics
# -------------------------
def _render_view_array(lh, rh, lh_rgba, rh_rgba, brain_opacity, cams, view,
                      df_all, used_set: set, patient_rgb, subd_col):
    pl = pv.Plotter(off_screen=True, window_size=C.WINDOW_SIZE)
    add_brain_mesh(pl, lh, lh_rgba, brain_opacity)
    add_brain_mesh(pl, rh, rh_rgba, brain_opacity)

    for _, r in df_all.iterrows():
        x,y,z = float(r["x"]), float(r["y"]), float(r["z"])
        elec = str(r["electrode"]).strip()
        rgb = patient_rgb if elec in used_set else C.UNUSED_GRAY_RGB

        rad = C.DEPTH_RADIUS
        if subd_col:
            try:
                if int(r[subd_col]) == 1: rad = C.SUBDURAL_RADIUS
            except Exception:
                pass

        sph = pv.Sphere(radius=rad, center=(x,y,z), theta_resolution=18, phi_resolution=18)
        pl.add_mesh(sph, color=rgb, opacity=C.ELECTRODE_OPACITY)

    pl.camera_position = cams[view]
    pl.reset_camera_clipping_range()
    img = pl.screenshot(transparent_background=C.TRANSPARENT_BG, scale=C.SS_SCALE, return_img=True)
    pl.close()
    return img


def render_patient_coverage_mosaic(pid: str, df_coords_all: pd.DataFrame, df_used: pd.DataFrame,
                                  lh, rh, out_png: Path, patient_colors,
                                  lh_rgba=None, rh_rgba=None, brain_opacity=C.BRAIN_OPACITY_CLEAN):
    pid = str(pid).strip()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df_all = df_coords_all.loc[df_coords_all["patient_id"] == pid].copy()
    if df_all.empty:
        raise RuntimeError(f"{pid}: no coords in df_coords_all")

    df_u = df_used.loc[df_used["patient_id"] == pid]
    used_set = set(df_u["electrode"].astype(str).str.strip().tolist())

    subd_col = first_present(df_all, C.SUBDURAL_COL_CANDIDATES)
    rgba = patient_colors[pid]
    patient_rgb = tuple(rgba[:3])

    bounds = (
        min(lh.bounds[0], rh.bounds[0]), max(lh.bounds[1], rh.bounds[1]),
        min(lh.bounds[2], rh.bounds[2]), max(lh.bounds[3], rh.bounds[3]),
        min(lh.bounds[4], rh.bounds[4]), max(lh.bounds[5], rh.bounds[5]),
    )
    cams = compute_cameras(bounds)

    imgs = [
        _render_view_array(lh, rh, lh_rgba, rh_rgba, brain_opacity, cams, v,
                           df_all, used_set, patient_rgb, subd_col)
        for v in C.PATIENT_COVERAGE_VIEWS
    ]
    iio.imwrite(out_png, np.hstack(imgs))
    print(f"[WROTE] {out_png}  (used={len(used_set)}, all={len(df_all)})")


# -------------------------
# Cell 2: render pipeline
# -------------------------
def render_atlas_figures(algo_name: Optional[str] = None) -> Dict[str, Path]:
    P = ensure_dirs(algo_name)

    df_meta = pd.read_csv(P["meta_out"], sep="\t")
    df_coords_all = pd.read_csv(P["coords_out"], sep="\t")
    df_coords_all.columns = [c.strip() for c in df_coords_all.columns]
    if "name" in df_coords_all.columns and "electrode" not in df_coords_all.columns:
        df_coords_all = df_coords_all.rename(columns={"name":"electrode"})

    df = merge_coords_and_meta(df_meta, df_coords_all)

    clusters = sorted(df["cluster"].astype(int).unique().tolist())
    patients = sorted(df["patient_id"].astype(str).unique().tolist())
    conds = sorted(df["condition"].astype(str).unique().tolist())

    print(f"[INFO] Unique clusters: {len(clusters)}")
    print(f"[INFO] Unique patients: {len(patients)}")
    print(f"[INFO] Unique conditions: {len(conds)} -> {conds}")

    cluster_colors = build_cluster_color_map(clusters)
    patient_colors = build_patient_colors_css4(patients)
    cond_colors = build_condition_colors_css4(conds)

    # save maps
    (P["render_out"]/ "cluster_color_map.json").write_text(json.dumps({str(k):list(v) for k,v in cluster_colors.items()}, indent=2))
    (P["render_out"]/ "patient_color_map.json").write_text(json.dumps({str(k):list(v) for k,v in patient_colors.items()}, indent=2))
    (P["render_out"]/ "condition_color_map.json").write_text(json.dumps({str(k):list(v) for k,v in cond_colors.items()}, indent=2))

    pv.set_jupyter_backend("static")
    try: pv.global_theme.multi_samples = 0
    except Exception: pass

    lh, rh = load_fsaverage_meshes()
    lh_ap, rh_ap = load_aparc_rgba()

    # --- A: clean ---
    if C.PLOT_ALL_CLUSTERS:
        out_all = P["render_out"] / "clean_brain" / "all_clusters"
        render_views(lh, rh, df, out_all, "clean_allclusters",
                     None, None, C.BRAIN_OPACITY_CLEAN, "cluster",
                     cluster_colors, patient_colors, cond_colors)
        legend_png({str(k):v for k,v in cluster_colors.items()}, "Cluster color legend", out_all/"legend_clusters.png", sort_key=lambda s:int(s))
        legend_png(patient_colors, "Patient color legend (global)", out_all/"legend_patients_global.png")
        legend_png(cond_colors, "Condition color legend (global)", out_all/"legend_conditions_global.png")

    if C.PLOT_ONE_CLUSTER_EACH:
        for cl in clusters:
            dfc = df.loc[df["cluster"] == cl].copy()
            base = P["render_out"] / "clean_brain" / f"cluster_{int(cl):02d}"
            render_views(lh, rh, dfc, base/"by_cluster", f"clean_cluster_{int(cl):02d}_bycluster",
                         None, None, C.BRAIN_OPACITY_CLEAN, "cluster", cluster_colors, patient_colors, cond_colors)
            if C.PLOT_PATIENT_COLORED_SUBFOLDERS_PER_CLUSTER:
                render_views(lh, rh, dfc, base/"by_patient", f"clean_cluster_{int(cl):02d}_bypatient",
                             None, None, C.BRAIN_OPACITY_CLEAN, "patient", cluster_colors, patient_colors, cond_colors)
            if C.PLOT_CONDITION_COLORED_SUBFOLDERS_PER_CLUSTER:
                render_views(lh, rh, dfc, base/"by_condition", f"clean_cluster_{int(cl):02d}_bycondition",
                             None, None, C.BRAIN_OPACITY_CLEAN, "condition", cluster_colors, patient_colors, cond_colors)

    # --- B: aparc ---
    if C.PLOT_ALL_CLUSTERS:
        out_all = P["render_out"] / "parcellated_aparc" / "all_clusters"
        render_views(lh, rh, df, out_all, "aparc_allclusters",
                     lh_ap, rh_ap, C.BRAIN_OPACITY_APARC, "cluster",
                     cluster_colors, patient_colors, cond_colors)
        legend_png({str(k):v for k,v in cluster_colors.items()}, "Cluster color legend", out_all/"legend_clusters.png", sort_key=lambda s:int(s))
        legend_png(patient_colors, "Patient color legend (global)", out_all/"legend_patients_global.png")
        legend_png(cond_colors, "Condition color legend (global)", out_all/"legend_conditions_global.png")

    if C.PLOT_ONE_CLUSTER_EACH:
        for cl in clusters:
            dfc = df.loc[df["cluster"] == cl].copy()
            base = P["render_out"] / "parcellated_aparc" / f"cluster_{int(cl):02d}"
            render_views(lh, rh, dfc, base/"by_cluster", f"aparc_cluster_{int(cl):02d}_bycluster",
                         lh_ap, rh_ap, C.BRAIN_OPACITY_APARC, "cluster", cluster_colors, patient_colors, cond_colors)
            if C.PLOT_PATIENT_COLORED_SUBFOLDERS_PER_CLUSTER:
                render_views(lh, rh, dfc, base/"by_patient", f"aparc_cluster_{int(cl):02d}_bypatient",
                             lh_ap, rh_ap, C.BRAIN_OPACITY_APARC, "patient", cluster_colors, patient_colors, cond_colors)
            if C.PLOT_CONDITION_COLORED_SUBFOLDERS_PER_CLUSTER:
                render_views(lh, rh, dfc, base/"by_condition", f"aparc_cluster_{int(cl):02d}_bycondition",
                             lh_ap, rh_ap, C.BRAIN_OPACITY_APARC, "condition", cluster_colors, patient_colors, cond_colors)

    # --- Patient coverage mosaics inside each cluster folder ---
    if C.WRITE_PATIENT_COVERAGE:
        cache = P["render_out"] / C.PATIENT_COVERAGE_CACHE_DIRNAME
        cache.mkdir(parents=True, exist_ok=True)

        def ensure_cached(pid: str) -> Path:
            p = cache / f"{pid}_coverage.png"
            if p.exists(): return p
            render_patient_coverage_mosaic(pid, df_coords_all, df, lh, rh, p, patient_colors,
                                          lh_rgba=None, rh_rgba=None, brain_opacity=C.BRAIN_OPACITY_CLEAN)
            return p

        for cl in clusters:
            dfc = df.loc[df["cluster"] == cl]
            pats = sorted(dfc["patient_id"].astype(str).unique().tolist())
            tdirs = [
                P["render_out"]/ "clean_brain"/ f"cluster_{int(cl):02d}"/ "patient_coverage",
                P["render_out"]/ "parcellated_aparc"/ f"cluster_{int(cl):02d}"/ "patient_coverage",
            ]
            for pid in pats:
                src = ensure_cached(pid)
                for t in tdirs:
                    t.mkdir(parents=True, exist_ok=True)
                    dst = t / f"{pid}_coverage.png"
                    if not dst.exists():
                        shutil.copyfile(src, dst)
                        print(f"[COPIED] {dst}")

    return P
