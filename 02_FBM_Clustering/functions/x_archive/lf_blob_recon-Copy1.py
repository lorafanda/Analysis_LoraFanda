# lf_blob_recon.py
from __future__ import annotations

import re, json, shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from matplotlib.backends.backend_pdf import PdfPages

import nibabel as nib
from nibabel.freesurfer.io import read_geometry, read_annot

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
from datetime import datetime

def _safe_tag(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s))
    return s.strip("_")

def _append_tsv_row(tsv_path: Path, row: dict) -> None:
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = (not tsv_path.exists())
    df = pd.DataFrame([row])
    df.to_csv(tsv_path, sep="\t", index=False, mode=("w" if header_needed else "a"), header=header_needed)


def save_cluster_mean_ersp_pngs(
    RUN_ID_230: str,
    RUN_ID_RECON: str,
    cluster_col_in_keep: str,
    *,
    fmax_hz: float = 500.0,
    vmin: float = -6.0,
    vmax: float = 6.0,
    use_median: bool = False,
    out_subdir: str = "cluster_mean_ersp",
):
    """
    Saves one ERSP summary image per cluster (mean or median) into:
      240/.../<RUN_ID_RECON>/reports/<out_subdir>/<cluster_col_in_keep>/

    Requires 230 run folder contains:
      - df_keep_with_clusters.parquet
      - ersp_keep.npy  (preferred stacked: (n, nF, nT))  OR ersp_keep.npz fallback
    """
    P = ensure_dirs(RUN_ID_230, RUN_ID_RECON)
    run230_dir = Path(P["run230_dir"])
    reports_dir = Path(P.get("reports_out", Path(P["run240_dir"]) / "reports"))
    out_dir = reports_dir / out_subdir / str(cluster_col_in_keep)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = Path(P["meta_in"])  # should point to df_keep_with_clusters.parquet
    df = pd.read_parquet(meta_path)

    if cluster_col_in_keep not in df.columns:
        raise ValueError(f"Cluster column not found: {cluster_col_in_keep}")

    # ---- Load ersp_keep (stacked preferred) ----
    f_npy = run230_dir / "ersp_keep.npy"
    f_npz = run230_dir / "ersp_keep.npz"

    if f_npy.exists():
        ersp = np.load(f_npy, allow_pickle=False)  # (n, nF, nT)
        if ersp.ndim != 3:
            raise ValueError(f"Expected ersp_keep.npy as (n,nF,nT), got shape={ersp.shape}")
    elif f_npz.exists():
        z = np.load(f_npz, allow_pickle=True)
        keys = sorted(z.files)
        ersp = np.stack([z[k].astype(np.float32, copy=False) for k in keys], axis=0)
    else:
        raise FileNotFoundError("Missing ersp_keep.npy / ersp_keep.npz in the 230 run folder.")

    if len(df) != ersp.shape[0]:
        raise ValueError(f"Row mismatch: df has {len(df)} rows but ersp has {ersp.shape[0]} samples.")

    labels = df[cluster_col_in_keep].to_numpy()
    finite = np.isfinite(labels)
    labels = labels[finite].astype(int)
    ersp = ersp[finite]

    uniq = np.unique(labels)
    nF, nT = ersp.shape[1], ersp.shape[2]

    # Axes: time in % and frequency in Hz (linear 0..fmax_hz)
    t_pct = np.linspace(0.0, 100.0, nT)
    f_hz  = np.linspace(0.0, float(fmax_hz), nF)
    half_x = 50.0  # vertical line at 50%

    agg_name = "median" if use_median else "mean"

    # Use 2-digit padding unless you have many clusters
    pad = max(2, len(str(int(np.max(uniq)))))

    for cid in uniq:
        idx = np.where(labels == int(cid))[0]
        if idx.size == 0:
            continue

        block = ersp[idx]  # (n_c, nF, nT)
        proto = np.median(block, axis=0) if use_median else np.mean(block, axis=0)

        fig, ax = plt.subplots(figsize=(2.6, 2.2), dpi=300)
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")

        # imshow with explicit extent => correct Hz / %time axes
        im = ax.imshow(
            proto,
            origin="lower",
            aspect="auto",
            cmap="bwr",
            vmin=vmin, vmax=vmax,
            extent=(t_pct[0], t_pct[-1], f_hz[0], f_hz[-1]),
            interpolation="nearest",
        )
        ax.axvline(half_x, color="gray", linewidth=1.0)

        ax.set_xlabel("Time (%)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"cl{cid} {agg_name} (n={idx.size})", fontsize=9)

        # minimal poster style
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        out_png = out_dir / f"{agg_name}_cluster_{cid:0{pad}d}.png"
        fig.savefig(out_png, transparent=True, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    return out_dir


def get_base_dir() -> Path:
    if C.BASE_DIR_OVERRIDE:
        return Path(C.BASE_DIR_OVERRIDE)
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

def _run230_dir(run_id_230: str) -> Path:
    return C.RUN230_ROOT / str(run_id_230).strip()


def _run240_dir(run_id_recon: str) -> Path:
    return C.RUN240_ROOT / str(run_id_recon).strip()


def ensure_dirs(run_id_230: str, run_id_recon: Optional[str] = None) -> Dict[str, Path]:
    """
    Resolve:
      - 230 run folder: outputs/230_blob_clustering_runs/<RUN_ID_230>/
      - 240 recon folder: outputs/240_blob_cluster_recon/<RUN_ID_RECON>/
    """
    rid230 = str(run_id_230).strip()
    ridrec = str(run_id_recon).strip() if run_id_recon else rid230

    in230 = _run230_dir(rid230)
    out240 = _run240_dir(ridrec)

    # Make recon output dirs
    out240.mkdir(parents=True, exist_ok=True)
    (out240 / C.INPUTS_SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)
    (out240 / C.ATLAS_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    (out240 / C.RENDER_OUT_DIRNAME).mkdir(parents=True, exist_ok=True)
    (out240 / C.REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    return {
        "base": get_base_dir(),
        "run_id_230": rid230,
        "run_id_recon": ridrec,

        # inputs (230)
        "run230_dir": in230,
        "meta_in": in230 / C.META_IN_KEEP_WITH_CLUSTERS,
        "X_in": in230 / C.X_IN_KEEP,

        # outputs (240)
        "run240_dir": out240,
        "inputs_snapshot": out240 / C.INPUTS_SNAPSHOT_DIR,
        "atlas_cache": out240 / C.ATLAS_CACHE_DIR,
        "meta_out": (out240 / C.ATLAS_CACHE_DIR) / C.META_OUT_NAME,
        "coords_out": (out240 / C.ATLAS_CACHE_DIR) / C.COORDS_OUT_NAME,
        "qc_out": (out240 / C.ATLAS_CACHE_DIR) / C.QC_OUT_NAME,
        "render_out": out240 / C.RENDER_OUT_DIRNAME,
        "reports_out": out240 / C.REPORTS_DIR,
    }


def first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lmap = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lmap:
            return lmap[cand.lower()]
    return None


# -------------------------
# Cluster column selection
# -------------------------
def pick_cluster_column(df: pd.DataFrame, cluster_col: Optional[str] = None) -> str:
    """
    If cluster_col provided -> use it.
    Else -> auto-detect exactly one column starting with 'cluster_'.
    """
    if cluster_col:
        cc = str(cluster_col).strip()
        if cc not in df.columns:
            raise ValueError(f"Requested cluster_col='{cc}' not found. Available: {list(df.columns)}")
        return cc

    cands = [c for c in df.columns if str(c).startswith(C.CLUSTER_COL_PREFIX)]
    if len(cands) == 0:
        raise ValueError(
            f"No cluster columns found. Expected at least one column starting with '{C.CLUSTER_COL_PREFIX}'. "
            f"Columns: {list(df.columns)}"
        )
    if len(cands) > 1:
        raise ValueError(
            f"Multiple cluster columns found: {cands}. "
            f"Pass cluster_col explicitly (e.g. cluster_col='{cands[0]}')."
        )
    return cands[0]


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
        raise ValueError(f"df_keep_with_clusters missing required columns: {sorted(miss)}")

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
    hits: List[Path] = []
    for b in bases:
        if not b.exists():
            continue
        hits += list(b.rglob("Lookup*.xlsx"))
        hits += list(b.rglob("*lookup*.xlsx"))
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
        xcol, ycol, zcol = C.LOOKUP_NATIVE_COLS
        space = "scannerRAS"
    elif C.ALLOW_EL_MNI_IF_NATIVE_MISSING and has_mni and df[C.LOOKUP_MNI_COLS[0]].notna().any():
        xcol, ycol, zcol = C.LOOKUP_MNI_COLS
        space = "MNI_like"
    else:
        raise FileNotFoundError(f"{pid}: Lookup.xlsx has no usable coords: {xlsx}")

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
    }).dropna(subset=["x", "y", "z", "electrode"])
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
            df = df.rename(columns={"name": "electrode"})
        df["patient_id"] = pid
        df["source_space"] = df.get("source_space", "tkrRAS")
        return df

    p_csv = paper1_contacts_csv(pid)
    if p_csv.exists():
        df = pd.read_csv(p_csv)
        df.columns = [c.strip() for c in df.columns]
        if "name" in df.columns and "electrode" not in df.columns:
            df = df.rename(columns={"name": "electrode"})
        df["patient_id"] = pid
        df["source_space"] = "tkrRAS"
        for c in ["x", "y", "z"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["electrode"] = df["electrode"].astype(str).str.strip()
        return df.dropna(subset=["x", "y", "z", "electrode"]).copy()

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
    for p in [subj_dir / "mri/brainmask.mgz", subj_dir / "mri/T1.mgz", subj_dir / "mri/orig.mgz"]:
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
        parts = [p for p in line.replace("\t", " ").split(" ") if p]
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


def subject_tkr_to_fsaverage_tkr(pid: str, points_tkr_subj: np.ndarray) -> Tuple[np.ndarray, Dict[str, str]]:
    subj_dir = patient_freesurfer_dir(pid)
    vox2ras_subj, vox2ras_tkr_subj, subj_mgz = load_mgz_matrices(subj_dir)
    tal_subj = parse_talairach_xfm(subj_dir / "mri/transforms/talairach.xfm")

    vox2ras_fs, vox2ras_tkr_fs, fs_mgz = load_mgz_matrices(C.FSAVERAGE_DIR)
    tal_fs = parse_talairach_xfm(C.FSAVERAGE_DIR / "mri/transforms/talairach.xfm")
    inv_tal_fs = np.linalg.inv(tal_fs)

    pts_scanner_subj = tkr_to_scanner(points_tkr_subj, vox2ras_subj, vox2ras_tkr_subj)
    pts_mni = apply_affine(pts_scanner_subj, tal_subj)
    pts_scanner_fs = apply_affine(pts_mni, inv_tal_fs)
    pts_tkr_fs = scanner_to_tkr(pts_scanner_fs, vox2ras_fs, vox2ras_tkr_fs)

    prov = {
        "pid": pid,
        "subj_dir": str(subj_dir),
        "subj_mgz_used": str(subj_mgz),
        "subj_tal": str(subj_dir / "mri/transforms/talairach.xfm"),
        "fs_mgz_used": str(fs_mgz),
        "fs_tal": str(C.FSAVERAGE_DIR / "mri/transforms/talairach.xfm"),
        "input_space": "subj_tkrRAS",
        "space_out": "fsaverage_tkrRAS_via_MNI305_affine",
    }
    return pts_tkr_fs, prov


def subject_scanner_to_fsaverage_tkr(pid: str, points_scanner_subj: np.ndarray) -> Tuple[np.ndarray, Dict[str, str]]:
    subj_dir = patient_freesurfer_dir(pid)
    tal_subj = parse_talairach_xfm(subj_dir / "mri/transforms/talairach.xfm")

    vox2ras_fs, vox2ras_tkr_fs, fs_mgz = load_mgz_matrices(C.FSAVERAGE_DIR)
    tal_fs = parse_talairach_xfm(C.FSAVERAGE_DIR / "mri/transforms/talairach.xfm")
    inv_tal_fs = np.linalg.inv(tal_fs)

    pts_mni = apply_affine(points_scanner_subj, tal_subj)
    pts_scanner_fs = apply_affine(pts_mni, inv_tal_fs)
    pts_tkr_fs = scanner_to_tkr(pts_scanner_fs, vox2ras_fs, vox2ras_tkr_fs)

    prov = {
        "pid": pid,
        "subj_dir": str(subj_dir),
        "subj_tal": str(subj_dir / "mri/transforms/talairach.xfm"),
        "fs_mgz_used": str(fs_mgz),
        "fs_tal": str(C.FSAVERAGE_DIR / "mri/transforms/talairach.xfm"),
        "input_space": "subj_scannerRAS",
        "space_out": "fsaverage_tkrRAS_via_MNI305_affine",
    }
    return pts_tkr_fs, prov


# -------------------------
# Cell 1: build atlas inputs (240 cache)
# -------------------------
def build_atlas_inputs(run_id_230: str, run_id_recon: Optional[str] = None, cluster_col: Optional[str] = None) -> Dict[str, Path]:
    P = ensure_dirs(run_id_230, run_id_recon)

    if not P["meta_in"].exists():
        raise FileNotFoundError(f"Missing 230 meta file: {P['meta_in']}")

    df_keep = pd.read_parquet(P["meta_in"])
    df_keep.columns = [c.strip() for c in df_keep.columns]

    use_cluster_col = pick_cluster_column(df_keep, cluster_col=cluster_col)

    # collapse sample-level -> contact-level (patient_id, electrode)
    df_contact = collapse_meta_to_contact_level(df_keep, cluster_col=use_cluster_col)
    df_contact.to_csv(P["meta_out"], sep="\t", index=False)

    # snapshot inputs for provenance
    snap = P["inputs_snapshot"] / C.META_IN_KEEP_WITH_CLUSTERS
    if not snap.exists():
        try:
            shutil.copyfile(P["meta_in"], snap)
        except Exception:
            # parquet copy can fail on some network perms; ignore if so
            pass

    patient_ids = sorted(df_contact["patient_id"].astype(str).str.strip().unique().tolist())
    qc_rows, all_rows = [], []

    for pid in patient_ids:
        try:
            subj_dir = patient_freesurfer_dir(pid)
            tal = subj_dir / "mri/transforms/talairach.xfm"
            if not tal.exists():
                raise FileNotFoundError(f"Missing talairach.xfm: {tal}")

            dfc = load_patient_contacts(pid)
            pts = dfc[["x", "y", "z"]].to_numpy(float)
            src = str(dfc.get("source_space", "tkrRAS").iloc[0]) if len(dfc) else "tkrRAS"

            if src.lower() == "tkrRAS".lower():
                pts_fs, prov = subject_tkr_to_fsaverage_tkr(pid, pts)
            elif src.lower() in ["scannerras", "subj_scannerras", "native", "native_scannerras"]:
                pts_fs, prov = subject_scanner_to_fsaverage_tkr(pid, pts)
            elif src.lower() == "mni_like":
                pts_fs, prov = subject_scanner_to_fsaverage_tkr(pid, pts)
            else:
                raise ValueError(f"Unknown source_space='{src}' for {pid}")

            out = dfc.copy()
            out["x"], out["y"], out["z"] = pts_fs[:, 0], pts_fs[:, 1], pts_fs[:, 2]
            out["coord_space"] = prov["space_out"]
            all_rows.append(out)

            qc_rows.append({
                "patient_id": pid,
                "n_contacts": int(len(out)),
                "status": "OK",
                "subj_dir": prov["subj_dir"],
                "input_space": prov["input_space"],
                "subj_tal": prov.get("subj_tal", ""),
                "fs_tal": prov["fs_tal"],
            })
            print(f"[OK] {pid}: transformed {len(out)} contacts -> fsaverage tkrRAS (input={prov['input_space']})")

        except Exception as e:
            qc_rows.append({
                "patient_id": pid,
                "n_contacts": 0,
                "status": f"ERROR: {e}",
                "subj_dir": str(patient_freesurfer_dir(pid)) if re.match(r"^(PAT_|MicroEPI|EL|el)", pid) else "",
            })
            print(f"[ERROR] {pid}: {e}")

    if not all_rows:
        raise RuntimeError("No patients transformed successfully; check QC output.")

    df_coords = pd.concat(all_rows, ignore_index=True)
    df_coords.columns = [c.strip() for c in df_coords.columns]
    if "name" in df_coords.columns and "electrode" not in df_coords.columns:
        df_coords = df_coords.rename(columns={"name": "electrode"})
    df_coords["patient_id"] = df_coords["patient_id"].astype(str).str.strip()
    df_coords["electrode"]  = df_coords["electrode"].astype(str).str.strip()
    df_coords.to_csv(P["coords_out"], sep="\t", index=False)

    pd.DataFrame(qc_rows).to_csv(P["qc_out"], sep="\t", index=False)

    print(f"[WROTE] {P['meta_out']}   (rows={len(df_contact)})")
    print(f"[WROTE] {P['coords_out']} (rows={len(df_coords)})")
    print(f"[WROTE] {P['qc_out']}")
    print(f"[INFO] Using cluster col: {use_cluster_col} -> exported as canonical 'cluster'")

    return P


# -------------------------
# Rendering: color maps
# -------------------------
def cohort_of(pid: str) -> str:
    pid = str(pid).strip()
    if pid.upper().startswith("EL"): return "EL"
    if pid.upper().startswith("PAT_"): return "PAT"
    if pid.startswith("MicroEPI"): return "MICRO"
    return "OTHER"


def build_patient_colors_css4(patients: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
    patients = sorted(set(map(str, patients)))
    pools = {"EL": C.EL_COLOR_NAMES, "PAT": C.PAT_COLOR_NAMES, "MICRO": C.MICRO_COLOR_NAMES}
    out: Dict[str, Tuple[float, float, float, float]] = {}

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

    # Ensure no duplicates
    seen = {}
    for p, rgba in out.items():
        key = tuple(round(x, 6) for x in rgba)
        if key in seen:
            raise ValueError(f"Duplicate patient color for {p} and {seen[key]} -> {rgba}")
        seen[key] = p

    return out


def build_condition_colors_css4(conditions: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
    conds = sorted(set(map(str, conditions)))
    if len(conds) > len(C.CONDITION_COLOR_NAMES):
        raise ValueError(f"Not enough condition colors: need {len(conds)}, have {len(C.CONDITION_COLOR_NAMES)}")
    out = {}
    for c, cname in zip(conds, C.CONDITION_COLOR_NAMES):
        if cname not in mcolors.CSS4_COLORS:
            raise ValueError(f"CSS4 color not found: {cname}")
        out[c] = mcolors.to_rgba(cname)
    return out


def build_cluster_color_map(clusters: List[int]) -> Dict[int, Tuple[float, float, float, float]]:
    # vibrant HSV spacing
    import colorsys
    cls = sorted(set(int(x) for x in clusters))
    n = max(1, len(cls))
    out = {}
    for i, cl in enumerate(cls):
        r, g, b = colorsys.hsv_to_rgb((i / n) % 1.0, 0.95, 0.98)
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
    df["condition"] = s.apply(lambda x: x.split(",")[0].strip() if "," in x else x.strip()).replace(
        {"": "NA", "nan": "NA", "None": "NA"}
    )
    return df


def merge_coords_and_meta(df_meta: pd.DataFrame, df_coords: pd.DataFrame) -> pd.DataFrame:
    df_meta = normalize_condition_column(df_meta)

    for df in (df_meta, df_coords):
        df["patient_id"] = df["patient_id"].astype(str).str.strip()
        df["electrode"]  = df["electrode"].astype(str).str.strip()

    df_meta["cluster"] = pd.to_numeric(df_meta["cluster"], errors="coerce")
    for c in ["x", "y", "z"]:
        df_coords[c] = pd.to_numeric(df_coords[c], errors="coerce")

    merged = df_coords.merge(
        df_meta[["patient_id", "electrode", "cluster", "condition"]],
        how="left",
        on=["patient_id", "electrode"],
        validate="m:1",
    )
    merged["matched"] = merged["cluster"].notna()

    print("\n[QC] Merge summary")
    print(f"  - coords rows:            {len(df_coords)}")
    print(f"  - missing cluster labels: {int((~merged['matched']).sum())}")
    print(f"  - missing any x/y/z:      {int(merged[['x','y','z']].isna().any(axis=1).sum())}")

    merged = merged.dropna(subset=["x", "y", "z", "cluster"]).copy()
    merged["cluster"] = merged["cluster"].astype(int)
    merged["condition"] = merged["condition"].astype(str).fillna("NA")

    subd = first_present(merged, C.SUBDURAL_COL_CANDIDATES)
    if subd:
        merged[subd] = pd.to_numeric(merged[subd], errors="coerce")
        before = len(merged)
        keep = np.zeros(len(merged), dtype=bool)
        if C.INCLUDE_DEPTH:
            keep |= (merged[subd] == 0)
        if C.INCLUDE_SUBDURAL:
            keep |= (merged[subd] == 1)
        merged = merged.loc[keep].copy()
        print(
            f"  - kept {len(merged)}/{before} using '{subd}' "
            f"with INCLUDE_DEPTH={C.INCLUDE_DEPTH}, INCLUDE_SUBDURAL={C.INCLUDE_SUBDURAL}"
        )

    return merged


def load_fsaverage_meshes() -> Tuple[pv.PolyData, pv.PolyData]:
    lh_path = C.FSAVERAGE_DIR / "surf" / "lh.pial"
    rh_path = C.FSAVERAGE_DIR / "surf" / "rh.pial"
    if not lh_path.exists() or not rh_path.exists():
        raise FileNotFoundError(f"Missing fsaverage pial: {lh_path} / {rh_path}")

    lh_v, lh_f = read_geometry(str(lh_path))
    rh_v, rh_f = read_geometry(str(rh_path))

    def to_pv(v, f):
        faces = np.hstack([np.full((f.shape[0], 1), 3, dtype=np.int64), f]).ravel()
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
        rgba = np.c_[rgb, np.full((rgb.shape[0], 1), 255, dtype=np.uint8)]
        lut = {int(lid): rgba[i] for i, lid in enumerate(label_ids)}
        out = np.zeros((labels.shape[0], 4), dtype=np.uint8)
        default = np.array([200, 200, 200, 255], dtype=np.uint8)
        for i, lid in enumerate(labels):
            out[i] = lut.get(int(lid), default)
        return out

    return annot_to_rgba(lh_annot), annot_to_rgba(rh_annot)


def compute_cameras(bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    size = max(xmax - xmin, ymax - ymin, zmax - zmin)
    d = 2.4 * size
    focal, upz = (cx, cy, cz), (0, 0, 1)
    return {
        "left": ((cx - d, cy, cz), focal, upz),
        "right": ((cx + d, cy, cz), focal, upz),
        "frontal": ((cx, cy + d, cz), focal, upz),
        "posterior": ((cx, cy - d, cz), focal, upz),
        "dorsal": ((cx, cy, cz + d), focal, (0, 1, 0)),
        "ventral": ((cx, cy, cz - d), focal, (0, 1, 0)),
    }


def add_brain_mesh(pl: pv.Plotter, mesh: pv.PolyData, rgba: Optional[np.ndarray], opacity: float):
    if rgba is None:
        pl.add_mesh(
            mesh, color=C.BRAIN_COLOR, opacity=opacity, smooth_shading=True,
            specular=C.BRAIN_SPECULAR, specular_power=C.BRAIN_SPECULAR_POWER,
            ambient=C.BRAIN_AMBIENT, diffuse=C.BRAIN_DIFFUSE
        )
    else:
        m = mesh.copy(deep=True)
        m.point_data["rgba"] = rgba
        pl.add_mesh(
            m, scalars="rgba", rgba=True, opacity=opacity, smooth_shading=True,
            specular=C.BRAIN_SPECULAR, specular_power=C.BRAIN_SPECULAR_POWER,
            ambient=C.BRAIN_AMBIENT, diffuse=C.BRAIN_DIFFUSE
        )


def add_electrodes(pl: pv.Plotter, df_plot: pd.DataFrame, color_by: str,
                  cluster_colors, patient_colors, cond_colors):
    subd = first_present(df_plot, C.SUBDURAL_COL_CANDIDATES)

    for _, r in df_plot.iterrows():
        x, y, z = float(r["x"]), float(r["y"]), float(r["z"])

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
                if int(r[subd]) == 1:
                    rad = C.SUBDURAL_RADIUS
            except Exception:
                pass

        sph = pv.Sphere(radius=rad, center=(x, y, z), theta_resolution=18, phi_resolution=18)
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
        out_png = activity_frame_path_241(
            run_id_230, run_id_recon,
            condition=condition,
            view=view,
            f_bins=(f0, f1),
            t_bins=(t0, t1),
        )
        pl.screenshot(str(out_png), transparent_background=C.TRANSPARENT_BG, scale=C.SS_SCALE)
        pl.close()
        print(f"[WROTE] {out_png}")


def legend_png(color_map: Dict[str, Tuple[float, float, float, float]], title: str, out_png: Path, sort_key=None):
    items = list(color_map.items())
    items = sorted(items, key=lambda kv: sort_key(kv[0]) if sort_key else str(kv[0]))
    labels = [str(k) for k, _ in items]
    colors = [v for _, v in items]
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
        x, y, z = float(r["x"]), float(r["y"]), float(r["z"])
        elec = str(r["electrode"]).strip()
        rgb = patient_rgb if elec in used_set else C.UNUSED_GRAY_RGB

        rad = C.DEPTH_RADIUS
        if subd_col:
            try:
                if int(r[subd_col]) == 1:
                    rad = C.SUBDURAL_RADIUS
            except Exception:
                pass

        sph = pv.Sphere(radius=rad, center=(x, y, z), theta_resolution=18, phi_resolution=18)
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
def render_atlas_figures(run_id_230: str, run_id_recon: Optional[str] = None) -> Dict[str, Path]:
    P = ensure_dirs(run_id_230, run_id_recon)

    if not P["meta_out"].exists() or not P["coords_out"].exists():
        raise FileNotFoundError(
            "Missing atlas cache files under 240. Run build_atlas_inputs() first.\n"
            f"Expected:\n  - {P['meta_out']}\n  - {P['coords_out']}"
        )

    df_meta = pd.read_csv(P["meta_out"], sep="\t")
    df_coords_all = pd.read_csv(P["coords_out"], sep="\t")
    df_coords_all.columns = [c.strip() for c in df_coords_all.columns]
    if "name" in df_coords_all.columns and "electrode" not in df_coords_all.columns:
        df_coords_all = df_coords_all.rename(columns={"name": "electrode"})

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

    # Save maps (in the recon render root)
    (P["render_out"] / "cluster_color_map.json").write_text(
        json.dumps({str(k): list(v) for k, v in cluster_colors.items()}, indent=2)
    )
    (P["render_out"] / "patient_color_map.json").write_text(
        json.dumps({str(k): list(v) for k, v in patient_colors.items()}, indent=2)
    )
    (P["render_out"] / "condition_color_map.json").write_text(
        json.dumps({str(k): list(v) for k, v in cond_colors.items()}, indent=2)
    )

    pv.set_jupyter_backend("static")
    try:
        pv.global_theme.multi_samples = 0
    except Exception:
        pass

    lh, rh = load_fsaverage_meshes()
    lh_ap, rh_ap = load_aparc_rgba()

    # --- A: clean brain ---
    if C.PLOT_ALL_CLUSTERS:
        out_all = P["render_out"] / "clean_brain" / "all_clusters"
        render_views(
            lh, rh, df, out_all, "clean_allclusters",
            None, None, C.BRAIN_OPACITY_CLEAN, "cluster",
            cluster_colors, patient_colors, cond_colors
        )
        legend_png({str(k): v for k, v in cluster_colors.items()}, "Cluster color legend",
                   out_all / "legend_clusters.png", sort_key=lambda s: int(s))
        legend_png(patient_colors, "Patient color legend (global)", out_all / "legend_patients_global.png")
        legend_png(cond_colors, "Condition color legend (global)", out_all / "legend_conditions_global.png")

    if C.PLOT_ONE_CLUSTER_EACH:
        for cl in clusters:
            dfc = df.loc[df["cluster"] == cl].copy()
            base = P["render_out"] / "clean_brain" / f"cluster_{int(cl):02d}"

            render_views(
                lh, rh, dfc, base / "by_cluster", f"clean_cluster_{int(cl):02d}_bycluster",
                None, None, C.BRAIN_OPACITY_CLEAN, "cluster",
                cluster_colors, patient_colors, cond_colors
            )

            if C.PLOT_PATIENT_COLORED_SUBFOLDERS_PER_CLUSTER:
                render_views(
                    lh, rh, dfc, base / "by_patient", f"clean_cluster_{int(cl):02d}_bypatient",
                    None, None, C.BRAIN_OPACITY_CLEAN, "patient",
                    cluster_colors, patient_colors, cond_colors
                )

            if C.PLOT_CONDITION_COLORED_SUBFOLDERS_PER_CLUSTER:
                render_views(
                    lh, rh, dfc, base / "by_condition", f"clean_cluster_{int(cl):02d}_bycondition",
                    None, None, C.BRAIN_OPACITY_CLEAN, "condition",
                    cluster_colors, patient_colors, cond_colors
                )

    # --- B: aparc brain ---
    if C.PLOT_ALL_CLUSTERS:
        out_all = P["render_out"] / "parcellated_aparc" / "all_clusters"
        render_views(
            lh, rh, df, out_all, "aparc_allclusters",
            lh_ap, rh_ap, C.BRAIN_OPACITY_APARC, "cluster",
            cluster_colors, patient_colors, cond_colors
        )
        legend_png({str(k): v for k, v in cluster_colors.items()}, "Cluster color legend",
                   out_all / "legend_clusters.png", sort_key=lambda s: int(s))
        legend_png(patient_colors, "Patient color legend (global)", out_all / "legend_patients_global.png")
        legend_png(cond_colors, "Condition color legend (global)", out_all / "legend_conditions_global.png")

    if C.PLOT_ONE_CLUSTER_EACH:
        for cl in clusters:
            dfc = df.loc[df["cluster"] == cl].copy()
            base = P["render_out"] / "parcellated_aparc" / f"cluster_{int(cl):02d}"

            render_views(
                lh, rh, dfc, base / "by_cluster", f"aparc_cluster_{int(cl):02d}_bycluster",
                lh_ap, rh_ap, C.BRAIN_OPACITY_APARC, "cluster",
                cluster_colors, patient_colors, cond_colors
            )

            if C.PLOT_PATIENT_COLORED_SUBFOLDERS_PER_CLUSTER:
                render_views(
                    lh, rh, dfc, base / "by_patient", f"aparc_cluster_{int(cl):02d}_bypatient",
                    lh_ap, rh_ap, C.BRAIN_OPACITY_APARC, "patient",
                    cluster_colors, patient_colors, cond_colors
                )

            if C.PLOT_CONDITION_COLORED_SUBFOLDERS_PER_CLUSTER:
                render_views(
                    lh, rh, dfc, base / "by_condition", f"aparc_cluster_{int(cl):02d}_bycondition",
                    lh_ap, rh_ap, C.BRAIN_OPACITY_APARC, "condition",
                    cluster_colors, patient_colors, cond_colors
                )

    # --- Patient coverage mosaics (copied into each cluster folder) ---
    if C.WRITE_PATIENT_COVERAGE:
        cache = P["render_out"] / C.PATIENT_COVERAGE_CACHE_DIRNAME
        cache.mkdir(parents=True, exist_ok=True)

        def ensure_cached(pid: str) -> Path:
            p = cache / f"{pid}_coverage.png"
            if p.exists():
                return p
            render_patient_coverage_mosaic(
                pid, df_coords_all, df, lh, rh, p, patient_colors,
                lh_rgba=None, rh_rgba=None, brain_opacity=C.BRAIN_OPACITY_CLEAN
            )
            return p

        for cl in clusters:
            dfc = df.loc[df["cluster"] == cl]
            pats = sorted(dfc["patient_id"].astype(str).unique().tolist())
            tdirs = [
                P["render_out"] / "clean_brain" / f"cluster_{int(cl):02d}" / "patient_coverage",
                P["render_out"] / "parcellated_aparc" / f"cluster_{int(cl):02d}" / "patient_coverage",
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

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None  # will error with a clear message if missing


# =========================
# Helpers: aparc label cache
# =========================

def _load_fsaverage_pial_and_aparc(fsaverage_dir: Path) -> Dict[str, object]:
    """
    Load fsaverage pial vertices and aparc annotations for lh/rh.
    Returns dict with:
        - lh_verts, rh_verts: (N,3) float
        - lh_aparc_names, rh_aparc_names: list[str] names per vertex (already decoded)
    """
    surf_lh = fsaverage_dir / "surf" / "lh.pial"
    surf_rh = fsaverage_dir / "surf" / "rh.pial"
    annot_lh = fsaverage_dir / "label" / "lh.aparc.annot"
    annot_rh = fsaverage_dir / "label" / "rh.aparc.annot"

    for p in [surf_lh, surf_rh, annot_lh, annot_rh]:
        if not p.exists():
            raise FileNotFoundError(f"Missing fsaverage dependency: {p}")

    lh_verts, _ = read_geometry(str(surf_lh))
    rh_verts, _ = read_geometry(str(surf_rh))

    lh_labels, lh_ctab, lh_names = read_annot(str(annot_lh))
    rh_labels, rh_ctab, rh_names = read_annot(str(annot_rh))

    lh_names = [n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n) for n in lh_names]
    rh_names = [n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n) for n in rh_names]

    # labels returned by read_annot are indices into names via the ctab mapping
    # nibabel returns "labels" as integer indices into names for FreeSurfer annot.
    # In practice: label values correspond to rows in ctab and index into names.
    # We'll map vertex->name via names[labels[i]].
    lh_vertex_names = [lh_names[int(i)] if int(i) < len(lh_names) else "unknown" for i in lh_labels]
    rh_vertex_names = [rh_names[int(i)] if int(i) < len(rh_names) else "unknown" for i in rh_labels]

    return dict(
        lh_verts=lh_verts,
        rh_verts=rh_verts,
        lh_vertex_names=np.array(lh_vertex_names, dtype=object),
        rh_vertex_names=np.array(rh_vertex_names, dtype=object),
    )


def compute_aparc_labels_cache(run_id_230: str, run_id_recon: str) -> Path:
    """
    Compute (patient_id, electrode) -> aparc label on fsaverage using nearest-vertex mapping.
    Saves to: <run240>/atlas_cache/electrode_aparc_labels_fsaverage.tsv
    Returns the output path.
    """
    from functions import lf_blob_recon_config as C  # local import to avoid circulars
    P = ensure_dirs(run_id_230, run_id_recon)

    coords_path = P["coords_out"]
    if not coords_path.exists():
        raise FileNotFoundError(f"Missing coords cache. Expected: {coords_path}")

    out_path = P["atlas_cache"] / C.APARC_LABELS_CACHE_NAME
    if out_path.exists():
        return out_path

    if KDTree is None:
        raise ImportError("scipy is required for aparc label assignment (scipy.spatial.cKDTree).")

    dfc = pd.read_csv(coords_path, sep="\t")
    required = {"patient_id", "electrode", "x", "y", "z"}
    missing = required - set(dfc.columns)
    if missing:
        raise ValueError(f"coords file missing columns: {sorted(missing)} in {coords_path}")

    fs = _load_fsaverage_pial_and_aparc(C.FSAVERAGE_DIR)

    lh_tree = KDTree(fs["lh_verts"])
    rh_tree = KDTree(fs["rh_verts"])

    pts = dfc[["x", "y", "z"]].to_numpy(dtype=float)

    d_lh, i_lh = lh_tree.query(pts, k=1)
    d_rh, i_rh = rh_tree.query(pts, k=1)

    use_lh = d_lh <= d_rh
    hemi = np.where(use_lh, "lh", "rh")
    vtx = np.where(use_lh, i_lh, i_rh).astype(int)
    dist = np.where(use_lh, d_lh, d_rh)

    aparc = np.empty(len(dfc), dtype=object)
    aparc[use_lh] = fs["lh_vertex_names"][i_lh[use_lh]]
    aparc[~use_lh] = fs["rh_vertex_names"][i_rh[~use_lh]]

    aparc_full = np.where(hemi == "lh", "lh-", "rh-").astype(object)
    aparc_full = (aparc_full + aparc.astype(str))

    out = pd.DataFrame({
        "patient_id": dfc["patient_id"].astype(str).to_numpy(),
        "electrode": dfc["electrode"].astype(str).to_numpy(),
        "hemi_fsavg": hemi,
        "nearest_vertex": vtx,
        "dist_to_pial_mm": np.asarray(dist, dtype=float),
        "aparc_label": aparc_full,
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)
    print(f"[WROTE] {out_path} (rows={len(out)})")
    return out_path


# =========================
# Helpers: heatmaps tables
# =========================

def _collapse_top_n(series_counts: pd.Series, top_n: int, other_name: str = "Other") -> pd.Series:
    """
    Keep only top_n categories; collapse the rest into `other_name`.
    Input series is counts indexed by category.
    """
    if top_n is None or top_n <= 0 or len(series_counts) <= top_n:
        return series_counts

    top = series_counts.sort_values(ascending=False).head(top_n)
    rest_sum = series_counts.drop(top.index, errors="ignore").sum()
    if rest_sum > 0:
        top.loc[other_name] = rest_sum
    return top


def build_cluster_matrices(
    run_id_230: str,
    run_id_recon: str,
    cluster_col_in_keep: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Returns dict of DataFrames:
      - M_region: clusters x aparc_label (proportions, contact-level)
      - M_patient: clusters x patient_id (proportions, contact-level)
      - M_condition: clusters x condition (proportions, sample-level from df_keep_with_clusters)
    """
    from functions import lf_blob_recon_config as C
    P = ensure_dirs(run_id_230, run_id_recon)

    # Contact-level meta with canonical 'cluster'
    df_meta = pd.read_csv(P["meta_out"], sep="\t")
    if "cluster" not in df_meta.columns:
        raise ValueError(f"meta_out missing canonical 'cluster' column: {P['meta_out']}")

    # Aparc labels cache
    aparc_path = compute_aparc_labels_cache(run_id_230, run_id_recon)
    df_aparc = pd.read_csv(aparc_path, sep="\t")

    # Merge at contact-level
    df_contact = df_meta.merge(df_aparc[["patient_id", "electrode", "aparc_label"]], on=["patient_id", "electrode"], how="left")
    df_contact["aparc_label"] = df_contact["aparc_label"].fillna("unknown")

    # Region matrix (contact-level): clusters x aparc
    ct = pd.crosstab(df_contact["cluster"], df_contact["aparc_label"])
    # keep global top regions
    region_totals = ct.sum(axis=0)
    keep_regions = _collapse_top_n(region_totals, C.HEATMAP_TOP_N_REGIONS, other_name="Other").index.tolist()
    # collapse cols outside keep into "Other"
    ct_region = ct.copy()
    if "Other" in keep_regions:
        other_cols = [c for c in ct_region.columns if c not in keep_regions or c == "Other"]
        # safer: sum columns not in keep_regions (excluding Other if not present)
        drop_cols = [c for c in ct_region.columns if c not in keep_regions and c != "Other"]
        if drop_cols:
            ct_region["Other"] = ct_region[drop_cols].sum(axis=1)
            ct_region = ct_region.drop(columns=drop_cols)
    # reorder columns
    ct_region = ct_region[[c for c in keep_regions if c in ct_region.columns]]
    M_region = ct_region.div(ct_region.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    # Patient matrix (contact-level): clusters x patient_id
    ct_pat = pd.crosstab(df_contact["cluster"], df_contact["patient_id"])
    pat_totals = ct_pat.sum(axis=0)
    keep_pats = _collapse_top_n(pat_totals, C.HEATMAP_TOP_N_PATIENTS, other_name="Other").index.tolist()
    ct_pat2 = ct_pat.copy()
    if "Other" in keep_pats:
        drop_cols = [c for c in ct_pat2.columns if c not in keep_pats and c != "Other"]
        if drop_cols:
            ct_pat2["Other"] = ct_pat2[drop_cols].sum(axis=1)
            ct_pat2 = ct_pat2.drop(columns=drop_cols)
    ct_pat2 = ct_pat2[[c for c in keep_pats if c in ct_pat2.columns]]
    M_patient = ct_pat2.div(ct_pat2.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    # Condition matrix (sample-level): clusters x condition
    df_keep_path = P["meta_in"]
    if not df_keep_path.exists():
        raise FileNotFoundError(f"Missing df_keep_with_clusters.parquet: {df_keep_path}")

    df_keep = pd.read_parquet(df_keep_path)

    # pick cluster column
    if cluster_col_in_keep is None:
        candidates = [c for c in df_keep.columns if c.startswith("cluster_")]
        if len(candidates) == 1:
            cluster_col_in_keep = candidates[0]
        else:
            raise ValueError(
                "cluster_col_in_keep not provided and df_keep has multiple cluster_ columns.\n"
                f"Candidates: {candidates}"
            )

    if cluster_col_in_keep not in df_keep.columns:
        raise ValueError(f"Requested cluster col not found in df_keep: {cluster_col_in_keep}")

    if "condition" not in df_keep.columns:
        raise ValueError("df_keep missing 'condition' column; cannot build cluster×condition heatmap.")

    ct_cond = pd.crosstab(df_keep[cluster_col_in_keep], df_keep["condition"])
    M_condition = ct_cond.div(ct_cond.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    # Make row index comparable across matrices: ensure clusters are sorted
    def _sort_rows(M: pd.DataFrame) -> pd.DataFrame:
        try:
            idx = M.index.astype(int)
            return M.loc[np.sort(idx)].copy()
        except Exception:
            return M.sort_index()

    return dict(
        M_region=_sort_rows(M_region),
        M_patient=_sort_rows(M_patient),
        M_condition=_sort_rows(M_condition),
        cluster_col_in_keep=cluster_col_in_keep,
    )


def _plot_heatmap(ax, M: pd.DataFrame, title: str, xlab: str, ylab: str) -> None:
    im = ax.imshow(M.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)

    ax.set_xticks(np.arange(M.shape[1]))
    ax.set_xticklabels(list(M.columns), rotation=90, fontsize=7)

    ax.set_yticks(np.arange(M.shape[0]))
    ax.set_yticklabels(list(M.index), fontsize=8)

    # colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.set_ylabel("Proportion", rotation=90, fontsize=9)


# =========================
# Main: PDF report
# =========================

def build_cluster_diagnostics_pdf(
    run_id_230: str,
    run_id_recon: str,
    cluster_col_in_keep: Optional[str] = None,
    *,
    ersp_proto_mode: str = "mean",   # "mean" or "median"
    fmax_hz: float = 500.0,
    vmin: float = -6.0,
    vmax: float = 6.0,
) -> Path:
    """
    Generates a single PDF report with:
      - Global heatmaps: cluster×region, cluster×condition, cluster×patient
      - Per-cluster pages embedding atlas renders (CLEAN ONLY) + ERSP prototype + small summaries.

    Output:
      <run240>/reports/cluster_diagnostics_report.pdf

    Also writes:
      <run240>/reports/cluster_ersp_prototypes/<cluster_col_in_keep>/{mean|median}_cluster_XX.png
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from pathlib import Path
    from typing import Optional, List, Tuple

    from functions import lf_blob_recon_config as C
    P = ensure_dirs(run_id_230, run_id_recon)

    # Require Cell 1 + Cell 2 outputs
    for req in [P["meta_out"], P["coords_out"]]:
        if not req.exists():
            raise FileNotFoundError(f"Missing atlas cache file: {req} (run build_atlas_inputs first)")

    if not P["render_out"].exists():
        raise FileNotFoundError(f"Missing render output dir: {P['render_out']} (run render_atlas_figures first)")

    mats = build_cluster_matrices(run_id_230, run_id_recon, cluster_col_in_keep=cluster_col_in_keep)
    M_region = mats["M_region"]
    M_condition = mats["M_condition"]
    M_patient = mats["M_patient"]
    cluster_col_in_keep = mats["cluster_col_in_keep"]

    # Load contact-level meta for per-cluster counts
    df_meta = pd.read_csv(P["meta_out"], sep="\t")
    clusters = sorted(df_meta["cluster"].unique().tolist(), key=lambda x: int(x) if str(x).isdigit() else str(x))

    out_pdf = P["reports_out"] / C.REPORT_PDF_NAME
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Building PDF: {out_pdf}")
    print(f"[INFO] Sample-level cluster col for condition matrix: {cluster_col_in_keep}")

    # -------------------------
    # NEW: load sample-level meta + ERSPs to compute per-cluster prototypes
    # -------------------------
    df_keep = pd.read_parquet(P["meta_in"])

    if cluster_col_in_keep not in df_keep.columns:
        raise ValueError(f"cluster_col_in_keep='{cluster_col_in_keep}' not in df_keep_with_clusters.parquet")

    # run230 directory (robust fallback)
    run230_dir = P.get("run230_dir", None)
    if run230_dir is None:
        # fallback to config root if ensure_dirs didn't provide it
        run230_dir = (C.RUN230_ROOT / run_id_230)
    run230_dir = Path(run230_dir)

    f_ersp_npy = run230_dir / "ersp_keep.npy"
    f_ersp_npz = run230_dir / "ersp_keep.npz"

    if f_ersp_npy.exists():
        ersp = np.load(f_ersp_npy, allow_pickle=False)  # (n, nF, nT)
        if ersp.ndim != 3:
            raise ValueError(f"Expected ersp_keep.npy as (n,nF,nT), got {ersp.shape}")
    elif f_ersp_npz.exists():
        z = np.load(f_ersp_npz, allow_pickle=True)
        keys = sorted(z.files)
        ersp = np.stack([z[k].astype(np.float32, copy=False) for k in keys], axis=0)
    else:
        raise FileNotFoundError(f"Missing ERSPs: {f_ersp_npy} or {f_ersp_npz}")

    if len(df_keep) != ersp.shape[0]:
        raise ValueError(f"Mismatch: df_keep rows={len(df_keep)} vs ersp samples={ersp.shape[0]}")

    # Prototype output dir
    proto_dir = P["reports_out"] / "cluster_ersp_prototypes" / str(cluster_col_in_keep)
    proto_dir.mkdir(parents=True, exist_ok=True)

    # axes mapping for ERSP plots
    nF, nT = ersp.shape[1], ersp.shape[2]
    t_pct = np.linspace(0.0, 100.0, nT)
    f_hz  = np.linspace(0.0, float(fmax_hz), nF)

    def _save_and_get_proto_png(cluster_int: int) -> Optional[Path]:
        """Compute and save ERSP prototype PNG for this cluster; return path or None if no samples."""
        m = df_keep[cluster_col_in_keep].astype("float").to_numpy()
        ok = np.isfinite(m)
        labs = m[ok].astype(int)
        e_ok = ersp[ok]

        idx = np.where(labs == int(cluster_int))[0]
        if idx.size == 0:
            return None

        block = e_ok[idx]  # (n_c, nF, nT)
        if ersp_proto_mode.lower().startswith("med"):
            proto = np.median(block, axis=0)
            tag = "median"
        else:
            proto = np.mean(block, axis=0)
            tag = "mean"

        # zero-pad cluster for nicer sorting
        pad = max(2, len(str(int(max([int(x) for x in clusters if str(x).isdigit()] + [cluster_int])))))
        out_png = proto_dir / f"{tag}_cluster_{cluster_int:0{pad}d}.png"

        # write only once
        if out_png.exists():
            return out_png

        fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=200)
        fig.patch.set_alpha(1.0)  # PDF page background; PNG is still saved with transparent=True below
        ax.set_facecolor("white")

        ax.imshow(
            proto,
            origin="lower",
            aspect="auto",
            cmap="bwr",
            vmin=vmin, vmax=vmax,
            extent=(t_pct[0], t_pct[-1], f_hz[0], f_hz[-1]),
            interpolation="nearest",
        )
        ax.axvline(50.0, color="gray", linewidth=1.0)

        ax.set_xlabel("Time (%)", fontsize=9)
        ax.set_ylabel("Frequency (Hz)", fontsize=9)
        ax.set_title(f"ERSP {tag} (n={idx.size})", fontsize=10)
        ax.tick_params(labelsize=8)

        fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.02, transparent=True)
        plt.close(fig)
        return out_png

    # -------------------------
    # FIX: find clean render paths that match your current renderer output
    # -------------------------
    def _find_clean_png(style_dir: Path, cluster_val, view: str) -> Optional[Path]:
        """
        Matches your current structure:
          <clean_brain>/cluster_05/by_cluster/clean_cluster_05_bycluster_left.png
        Also includes a robust rglob fallback.
        """
        try:
            k_int = int(cluster_val)
        except Exception:
            return None

        # try both non-padded and padded; prefer padded (your renderer uses padded)
        k2 = f"{k_int:02d}"
        candidates = [
            style_dir / f"cluster_{k2}" / "by_cluster" / f"clean_cluster_{k2}_bycluster_{view}.png",
            style_dir / f"cluster_{k_int}" / "by_cluster" / f"clean_cluster_{k_int}_bycluster_{view}.png",
            style_dir / f"cluster_{k2}" / f"clean_cluster_{k2}_bycluster_{view}.png",
            style_dir / f"cluster_{k_int}" / f"clean_cluster_{k_int}_bycluster_{view}.png",
        ]
        for p in candidates:
            if p.exists():
                return p

        # fallback: search by tokens
        hits = list(style_dir.rglob(f"*cluster*{k_int}*{view}*.png"))
        if hits:
            # deterministic pick
            hits = sorted(hits, key=lambda x: str(x))
            return hits[0]
        return None

    # -------------------------
    # PDF generation
    # -------------------------
    with PdfPages(out_pdf) as pdf:
        # --- Page 1: global heatmaps ---
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 0.8, 1.0], hspace=0.5)

        ax1 = fig.add_subplot(gs[0, 0])
        _plot_heatmap(ax1, M_region, "Clusters × aparc regions (contact-level)", "aparc_label", "cluster")

        ax2 = fig.add_subplot(gs[1, 0])
        _plot_heatmap(ax2, M_condition, "Clusters × condition (sample-level)", "condition", f"cluster ({cluster_col_in_keep})")

        ax3 = fig.add_subplot(gs[2, 0])
        _plot_heatmap(ax3, M_patient, "Clusters × patient (contact-level)", "patient_id", "cluster")

        fig.suptitle(f"Cluster diagnostics — run230={run_id_230} — run240={run_id_recon}", fontsize=13, y=0.995)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Per-cluster pages ---
        for k in clusters:
            df_k = df_meta[df_meta["cluster"] == k]
            n_contacts = len(df_k)
            n_pat = df_k["patient_id"].nunique()

            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.7], hspace=0.35, wspace=0.25)

            # 2×3 image grid (top 2 rows).
            img_slots = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]

            # CLEAN ONLY
            clean_dir = P["render_out"] / "clean_brain"

            # Use up to 5 views + 1 ERSP prototype in the last slot (poster-friendly)
            view_list = list(C.PDF_VIEWS)[:5]
            proto_slot = img_slots[-1]

            # Plot the 5 clean renders
            for (slot, view) in zip(img_slots[:-1], view_list):
                ax = fig.add_subplot(gs[slot[0], slot[1]])
                ax.axis("off")

                png = _find_clean_png(clean_dir, k, view)
                if png is None:
                    ax.text(0.5, 0.5, f"Missing CLEAN render\ncluster {k}\nview {view}",
                            ha="center", va="center", fontsize=10)
                    continue

                img = plt.imread(str(png))
                ax.imshow(img)
                ax.set_title(f"clean • {view}", fontsize=10)

            # Plot ERSP prototype (and save it)
            axp_img = fig.add_subplot(gs[proto_slot[0], proto_slot[1]])
            axp_img.axis("off")

            try:
                k_int = int(k)
            except Exception:
                k_int = None

            proto_png = _save_and_get_proto_png(k_int) if k_int is not None else None
            if proto_png is None or not proto_png.exists():
                axp_img.text(0.5, 0.5, f"Missing ERSP prototype\ncluster {k}",
                             ha="center", va="center", fontsize=10)
            else:
                img = plt.imread(str(proto_png))
                axp_img.imshow(img)
                axp_img.set_title(f"ERSP {ersp_proto_mode}", fontsize=10)

            # Bottom row: mini summaries from matrices
            axr = fig.add_subplot(gs[2, 0])
            axc = fig.add_subplot(gs[2, 1])
            axp = fig.add_subplot(gs[2, 2])

            # Region top 10 for this cluster
            if k in M_region.index:
                s = M_region.loc[k].sort_values(ascending=False).head(10)
                axr.bar(np.arange(len(s)), s.to_numpy())
                axr.set_xticks(np.arange(len(s)))
                axr.set_xticklabels(list(s.index), rotation=60, ha="right", fontsize=8)
                axr.set_title("Top regions", fontsize=10)
                axr.set_ylim(0, max(0.05, float(s.max()) * 1.1))
            else:
                axr.text(0.5, 0.5, "No region row", ha="center", va="center")

            # Condition proportions for this cluster (sample-level)
            try:
                k_int = int(k)
            except Exception:
                k_int = k

            if k_int in M_condition.index:
                s = M_condition.loc[k_int]
                axc.bar(np.arange(len(s)), s.to_numpy())
                axc.set_xticks(np.arange(len(s)))
                axc.set_xticklabels(list(s.index), rotation=0, fontsize=9)
                axc.set_title("Conditions", fontsize=10)
                axc.set_ylim(0, max(0.05, float(s.max()) * 1.1))
            else:
                axc.text(0.5, 0.5, "No condition row", ha="center", va="center")

            # Patient top 10
            if k in M_patient.index:
                s = M_patient.loc[k].sort_values(ascending=False).head(10)
                axp.bar(np.arange(len(s)), s.to_numpy())
                axp.set_xticks(np.arange(len(s)))
                axp.set_xticklabels(list(s.index), rotation=60, ha="right", fontsize=8)
                axp.set_title("Top patients", fontsize=10)
                axp.set_ylim(0, max(0.05, float(s.max()) * 1.1))
            else:
                axp.text(0.5, 0.5, "No patient row", ha="center", va="center")

            fig.suptitle(f"Cluster {k} — contacts={n_contacts} — patients={n_pat}", fontsize=13, y=0.995)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[WROTE] {out_pdf}")
    print(f"[WROTE] ERSP prototypes: {proto_dir}")
    return out_pdf



####################################
# For: Atlas Activty Visualization: 
####################################

# =========================
# Surface ERSP maps (activity + density overlay)
# =========================


try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

def _load_ersp_keep_from_run230(run230_dir: Path) -> np.ndarray:
    """
    Load stacked ERSPs aligned with df_keep rows:
      - ersp_keep.npy: (n_samples, nF, nT) float32
      - OR ersp_keep.npz: keys -> arrays (nF,nT), stacked by sorted key order
    """
    f_npy = run230_dir / "ersp_keep.npy"
    f_npz = run230_dir / "ersp_keep.npz"

    if f_npy.exists():
        ersp = np.load(f_npy, allow_pickle=False)
        if ersp.ndim != 3:
            raise ValueError(f"Expected ersp_keep.npy as (n,nF,nT), got shape={ersp.shape}")
        return ersp

    if f_npz.exists():
        z = np.load(f_npz, allow_pickle=True)
        keys = sorted(z.files)
        ersp = np.stack([z[k].astype(np.float32, copy=False) for k in keys], axis=0)
        return ersp

    raise FileNotFoundError(f"Missing {f_npy} and {f_npz} in 230 run folder.")


def _norm_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace({"nan": "NA", "None": "NA", "": "NA"})


def _pick_condition_col(df: pd.DataFrame) -> str:
    ccol = first_present(df, C.CONDITION_COL_CANDIDATES)
    return ccol if ccol else "condition"


def _compute_contact_values_for_bin(
    df_keep: pd.DataFrame,
    ersp: np.ndarray,
    *,
    f_bin: int,
    t_bin: int,
    condition: Optional[str] = None,
    agg_within_contact: str = "mean",   # "mean" or "median"
) -> pd.DataFrame:
    """
    Returns contact-level values with columns:
      patient_id, electrode, value

    - Optionally filters samples by a condition (sample-level).
    - Aggregates multiple samples per contact via mean/median.

    Notes:
      df_keep must have patient_id, electrode, and (if filtering) a condition column.
      ersp must align row-wise with df_keep.
    """
    if len(df_keep) != ersp.shape[0]:
        raise ValueError(f"Row mismatch: df_keep={len(df_keep)} vs ersp={ersp.shape[0]}")

    if "patient_id" not in df_keep.columns or "electrode" not in df_keep.columns:
        raise ValueError("df_keep must contain columns: patient_id, electrode")

    if not (0 <= f_bin < ersp.shape[1]) or not (0 <= t_bin < ersp.shape[2]):
        raise ValueError(f"Requested bin out of range: f_bin={f_bin}, t_bin={t_bin}, ersp shape={ersp.shape}")

    df = df_keep[["patient_id", "electrode"]].copy()
    df["patient_id"] = _norm_str_series(df["patient_id"])
    df["electrode"]  = _norm_str_series(df["electrode"])

    # sample value at bin
    v = ersp[:, f_bin, t_bin].astype(np.float32, copy=False)
    df["value"] = v

    # optional condition filter
    if condition is not None:
        ccol = _pick_condition_col(df_keep)
        if ccol not in df_keep.columns:
            raise ValueError(f"Condition filter requested but no condition column found. Tried: {C.CONDITION_COL_CANDIDATES}")
        cond = _norm_str_series(df_keep[ccol])
        df = df.loc[cond == str(condition).strip()].copy()

    # aggregate to contact-level
    if agg_within_contact.lower().startswith("med"):
        g = df.groupby(["patient_id", "electrode"], as_index=False)["value"].median()
    else:
        g = df.groupby(["patient_id", "electrode"], as_index=False)["value"].mean()

    return g


def _gaussian_weights(d: np.ndarray, sigma_mm: float) -> np.ndarray:
    # d in mm
    return np.exp(-(d * d) / (2.0 * sigma_mm * sigma_mm))


def _project_contacts_to_surface(
    df_vals: pd.DataFrame,
    df_coords_fsavg: pd.DataFrame,
    *,
    lh_pts: np.ndarray,
    rh_pts: np.ndarray,
    radius_mm: float = 12.0,
    sigma_mm: float = 5.0,
    exclude_contacts_dist_to_pial_mm_gt: Optional[float] = 12.0,
) -> Dict[str, np.ndarray]:
    """
    Spatial interpolation onto pial surface vertices using a Gaussian kernel.

    Returns dict with:
      act_lh, den_lh, act_rh, den_rh (float arrays per vertex; act is weighted mean)
      plus: wsum_lh, wsum_rh (raw density weights)
    """
    if KDTree is None:
        raise ImportError("scipy is required for surface projection (scipy.spatial.cKDTree).")

    need = {"patient_id", "electrode", "x", "y", "z"}
    miss = need - set(df_coords_fsavg.columns)
    if miss:
        raise ValueError(f"coords_out missing columns: {sorted(miss)}")

    coords = df_coords_fsavg.copy()
    coords["patient_id"] = _norm_str_series(coords["patient_id"])
    coords["electrode"]  = _norm_str_series(coords["electrode"])
    for c in ["x", "y", "z"]:
        coords[c] = pd.to_numeric(coords[c], errors="coerce")
    coords = coords.dropna(subset=["x", "y", "z", "patient_id", "electrode"]).copy()

    # merge values into coords
    vals = df_vals.copy()
    vals["patient_id"] = _norm_str_series(vals["patient_id"])
    vals["electrode"]  = _norm_str_series(vals["electrode"])

    m = coords.merge(vals, on=["patient_id", "electrode"], how="inner")
    if m.empty:
        raise RuntimeError("No overlap between df_vals and coords_out (patient_id,electrode).")

    # optional: exclude contacts far from pial (best practice for depth-heavy cohorts)
    # We approximate pial distance by nearest-vertex distance (computed on the fly).
    lh_tree = KDTree(lh_pts)
    rh_tree = KDTree(rh_pts)

    pts = m[["x", "y", "z"]].to_numpy(float)
    d_lh, _ = lh_tree.query(pts, k=1)
    d_rh, _ = rh_tree.query(pts, k=1)
    dmin = np.minimum(d_lh, d_rh)

    m["dist_to_pial_mm"] = dmin
    if exclude_contacts_dist_to_pial_mm_gt is not None:
        thr = float(exclude_contacts_dist_to_pial_mm_gt)
        before = len(m)
        m = m.loc[m["dist_to_pial_mm"] <= thr].copy()
        print(f"[INFO] Excluding contacts with dist_to_pial_mm>{thr}: kept {len(m)}/{before}")
        if m.empty:
            raise RuntimeError("All contacts were excluded by dist_to_pial threshold; relax the threshold.")

    # Build per-hemi vertex KDTree for neighborhood queries
    lh_tree = KDTree(lh_pts)
    rh_tree = KDTree(rh_pts)

    act_num_lh = np.zeros(len(lh_pts), dtype=np.float64)
    wsum_lh    = np.zeros(len(lh_pts), dtype=np.float64)

    act_num_rh = np.zeros(len(rh_pts), dtype=np.float64)
    wsum_rh    = np.zeros(len(rh_pts), dtype=np.float64)

    # Assign each contact to the closer hemi by nearest-vertex distance
    pts = m[["x", "y", "z"]].to_numpy(float)
    vval = m["value"].to_numpy(float)

    d_lh, _ = lh_tree.query(pts, k=1)
    d_rh, _ = rh_tree.query(pts, k=1)
    use_lh = d_lh <= d_rh

    # For each contact: add a Gaussian “bump” to vertices within radius_mm
    rad = float(radius_mm)
    sig = float(sigma_mm)

    # LH
    idx_lh = np.where(use_lh)[0]
    for i in idx_lh:
        p = pts[i]
        val = float(vval[i])
        nbrs = lh_tree.query_ball_point(p, r=rad)
        if not nbrs:
            continue
        dv = lh_pts[nbrs] - p[None, :]
        d = np.sqrt((dv * dv).sum(axis=1))
        w = _gaussian_weights(d, sig)
        act_num_lh[nbrs] += w * val
        wsum_lh[nbrs]    += w

    # RH
    idx_rh = np.where(~use_lh)[0]
    for i in idx_rh:
        p = pts[i]
        val = float(vval[i])
        nbrs = rh_tree.query_ball_point(p, r=rad)
        if not nbrs:
            continue
        dv = rh_pts[nbrs] - p[None, :]
        d = np.sqrt((dv * dv).sum(axis=1))
        w = _gaussian_weights(d, sig)
        act_num_rh[nbrs] += w * val
        wsum_rh[nbrs]    += w

    # Weighted mean (avoid divide-by-zero)
    eps = 1e-12
    act_lh = act_num_lh / np.maximum(wsum_lh, eps)
    act_rh = act_num_rh / np.maximum(wsum_rh, eps)

    return dict(
        act_lh=act_lh.astype(np.float32),
        den_lh=wsum_lh.astype(np.float32),
        act_rh=act_rh.astype(np.float32),
        den_rh=wsum_rh.astype(np.float32),
        wsum_lh=wsum_lh.astype(np.float32),
        wsum_rh=wsum_rh.astype(np.float32),
    )


def _rgba_from_activity(values: np.ndarray, *, vmin: float, vmax: float, cmap_name: str = "bwr") -> np.ndarray:
    cmap = mpl.cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = cmap(norm(values))  # float 0..1
    return (rgba * 255).astype(np.uint8)


def _rgba_from_density_gray(density: np.ndarray, *, gamma: float = 0.6) -> np.ndarray:
    # Normalize to [0,1] (robust: use max)
    d = np.asarray(density, float)
    dmax = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
    if dmax <= 0:
        g = np.zeros_like(d, dtype=np.float32)
    else:
        g = (d / dmax).astype(np.float32)
    # gamma < 1 boosts low densities for visibility
    g = np.power(np.clip(g, 0, 1), float(gamma))
    rgb = (g[:, None] * 255.0).astype(np.uint8)
    a = np.full((len(g), 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, rgb, rgb, a], axis=1)  # grayscale RGBA


def _blend_rgba(activity_rgba: np.ndarray, density_gray_rgba: np.ndarray, *, mix: float = 0.25) -> np.ndarray:
    """
    Blend density grayscale onto activity:
      out = (1-mix)*activity + mix*density_gray
    mix=0 -> pure activity
    mix=1 -> pure density gray
    """
    m = float(np.clip(mix, 0.0, 1.0))
    a = activity_rgba.astype(np.float32)
    g = density_gray_rgba.astype(np.float32)
    out = (1.0 - m) * a + m * g
    out[:, 3] = 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def render_surface_activity_maps_for_bin(
    RUN_ID_230: str,
    RUN_ID_RECON: str,
    *,
    f_bin: int,
    t_bin: int,
    condition: Optional[str] = None,
    agg_within_contact: str = "mean",
    vmin: float = -6.0,
    vmax: float = 6.0,
    density_gray: bool = True,
    density_mix: float = 0.25,
    density_gamma: float = 0.6,
    radius_mm: float = 12.0,
    sigma_mm: float = 5.0,
    exclude_contacts_dist_to_pial_mm_gt: Optional[float] = 12.0,
    out_tag: Optional[str] = None,
) -> Path:
    """
    Produces pial surface renders where vertex colors encode activity (bwr[-6,6]),
    optionally blended with density grayscale (black->white).

    Outputs:
      <240>/<RUN_ID_RECON>/reports/surface_activity_maps/<tag>_<view>.png

    Returns: output directory path.
    """
    P = ensure_dirs(RUN_ID_230, RUN_ID_RECON)

    # Load inputs
    df_keep = pd.read_parquet(P["meta_in"])
    df_coords = pd.read_csv(P["coords_out"], sep="\t")
    df_coords.columns = [c.strip() for c in df_coords.columns]
    if "name" in df_coords.columns and "electrode" not in df_coords.columns:
        df_coords = df_coords.rename(columns={"name": "electrode"})

    ersp = _load_ersp_keep_from_run230(Path(P["run230_dir"]))

    # Contact values for this bin
    df_vals = _compute_contact_values_for_bin(
        df_keep, ersp,
        f_bin=int(f_bin), t_bin=int(t_bin),
        condition=condition,
        agg_within_contact=agg_within_contact,
    )

    # Load meshes + vertices
    lh_mesh, rh_mesh = load_fsaverage_meshes()
    lh_pts = np.asarray(lh_mesh.points, dtype=np.float32)
    rh_pts = np.asarray(rh_mesh.points, dtype=np.float32)

    # Project to surface
    fields = _project_contacts_to_surface(
        df_vals, df_coords,
        lh_pts=lh_pts, rh_pts=rh_pts,
        radius_mm=radius_mm, sigma_mm=sigma_mm,
        exclude_contacts_dist_to_pial_mm_gt=exclude_contacts_dist_to_pial_mm_gt,
    )

    # RGBA
    act_rgba_lh = _rgba_from_activity(fields["act_lh"], vmin=vmin, vmax=vmax, cmap_name="bwr")
    act_rgba_rh = _rgba_from_activity(fields["act_rh"], vmin=vmin, vmax=vmax, cmap_name="bwr")

    if density_gray:
        den_rgba_lh = _rgba_from_density_gray(fields["den_lh"], gamma=density_gamma)
        den_rgba_rh = _rgba_from_density_gray(fields["den_rh"], gamma=density_gamma)
        act_rgba_lh = _blend_rgba(act_rgba_lh, den_rgba_lh, mix=density_mix)
        act_rgba_rh = _blend_rgba(act_rgba_rh, den_rgba_rh, mix=density_mix)

    # Output naming
    cond_tag = "all" if condition is None else str(condition).strip()
    tag = out_tag or f"surf_f{int(f_bin):03d}_t{int(t_bin):03d}_cond-{cond_tag}"
    out_dir = Path(P["reports_out"]) / "surface_activity_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Render views
    pv.set_jupyter_backend("static")
    try:
        pv.global_theme.multi_samples = 0
    except Exception:
        pass

    bounds = (
        min(lh_mesh.bounds[0], rh_mesh.bounds[0]), max(lh_mesh.bounds[1], rh_mesh.bounds[1]),
        min(lh_mesh.bounds[2], rh_mesh.bounds[2]), max(lh_mesh.bounds[3], rh_mesh.bounds[3]),
        min(lh_mesh.bounds[4], rh_mesh.bounds[4]), max(lh_mesh.bounds[5], rh_mesh.bounds[5]),
    )
    cams = compute_cameras(bounds)

    for view in C.VIEWS_TO_SAVE:
        pl = pv.Plotter(off_screen=True, window_size=C.WINDOW_SIZE)

        # Color the surface directly (no electrode spheres)
        add_brain_mesh(pl, lh_mesh, act_rgba_lh, opacity=1.0)
        add_brain_mesh(pl, rh_mesh, act_rgba_rh, opacity=1.0)

        pl.camera_position = cams[view]
        pl.reset_camera_clipping_range()

        out_png = out_dir / f"{tag}_{view}.png"
        pl.screenshot(str(out_png), transparent_background=C.TRANSPARENT_BG, scale=C.SS_SCALE)
        pl.close()
        print(f"[WROTE] {out_png}")

    # Cache the numeric fields (so you can re-render without recomputing)
    cache_dir = Path(P["atlas_cache"]) / "surface_activity_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_dir / f"{tag}_fields.npz",
        act_lh=fields["act_lh"], den_lh=fields["den_lh"],
        act_rh=fields["act_rh"], den_rh=fields["den_rh"],
        params=dict(
            f_bin=int(f_bin), t_bin=int(t_bin), condition=cond_tag,
            vmin=float(vmin), vmax=float(vmax),
            density_gray=bool(density_gray), density_mix=float(density_mix), density_gamma=float(density_gamma),
            radius_mm=float(radius_mm), sigma_mm=float(sigma_mm),
            exclude_contacts_dist_to_pial_mm_gt=None if exclude_contacts_dist_to_pial_mm_gt is None else float(exclude_contacts_dist_to_pial_mm_gt),
        ),
    )
    print(f"[WROTE] {cache_dir / f'{tag}_fields.npz'}")

    return out_dir



# ---- add to lf_blob_recon.py ----
from typing import Optional, Tuple, Dict, Any
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

import pyvista as pv

import functions.lf_blob_recon_config as C


def render_surface_activity_from_ersp_window(
    run_id_230: str,
    run_id_recon: str,
    *,
    f_bins: tuple,                 # (f0, f1) INCLUSIVE bin indices
    t_bins: tuple,                 # (t0, t1) INCLUSIVE bin indices
    condition: str = None,         # e.g. "picture" or None for all
    agg_contact: str = "mean",     # "mean" | "median" | "maxabs"
    sigma_mm: float = 4.0,
    k_nearest: int = 4,
    vmin: float = -6.0,
    vmax: float = 6.0,
    cmap_name: str = "bwr",
    density_saturation: bool = True,
    s_min: float = 0.15,
    s_max: float = 1.25,
    density_gamma: float = 0.6,
    density_norm_q: float = 0.98,
    exclude_contacts_dist_to_pial_mm_gt: float = 12.0,
    views: list = None,            # e.g. ["left","right","dorsal","ventral"]
    out_subdir: str = "surface_activity",
    density_alpha: bool = True,
    alpha_min: float = 0.10,
    alpha_max: float = 1.00,
    alpha_gamma: float = 0.70,
    base_brain_gray_rgb: tuple = (0.22, 0.22, 0.22),
):
    """
    What it DOES:
      - Loads ERSP stack from 230/<RUN_ID_230>/ersp_keep.npy (or .npz fallback).
      - Loads a sample-level meta parquet from 230/<RUN_ID_230>/ that MATCHES the ERSP row count
        (auto-picks between df_keep_with_clusters.parquet, df_keep.parquet, etc.).
      - Optionally filters samples by `condition`.
      - Computes one scalar per sample = mean ERSP in the (f_bins,t_bins) window.
      - Aggregates to contact-level (patient_id,electrode) using agg_contact (mean/median/maxabs).
      - Merges with 240/<RUN_ID_RECON>/atlas_cache/electrode_coords_fsaverage_tkr.tsv for xyz.
      - Excludes contacts far from the fsaverage pial (> exclude_contacts_dist_to_pial_mm_gt).
      - Projects contact scalars to the *pial surface vertices* using kNN + Gaussian weights (sigma_mm).
      - Renders the surface colored by activity (bwr, vmin/vmax). Optionally darkens low-density areas.

    What it DOES NOT do:
      - It is NOT volumetric. Deep contacts far from cortex are either downweighted or excluded.
      - It does NOT “open” sulci or do cortical flattening; it renders fsaverage pial views.
      - It does NOT guarantee anatomical correctness for deep SEEG trajectories (surface projection is an approximation).
      - It does NOT fix a fundamentally wrong meta↔ERSP pairing; it only auto-selects a matching meta file by row count.
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path

    try:
        from scipy.spatial import cKDTree as KDTree
    except Exception as e:
        raise ImportError("scipy is required (scipy.spatial.cKDTree).") from e

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pyvista as pv
    from nibabel.freesurfer.io import read_geometry

    # -------------------------
    # Resolve paths
    # -------------------------
    P = ensure_dirs(run_id_230, run_id_recon)
    run230_dir = Path(P["run230_dir"])
    coords_path = Path(P["coords_out"])  # 240 atlas_cache electrode_coords_fsaverage_tkr.tsv
    if not coords_path.exists():
        raise FileNotFoundError(f"Missing coords cache under 240: {coords_path} (run build_atlas_inputs first)")

    # -------------------------
    # Load ERSP stack (n, nF, nT)
    # -------------------------
    f_npy = run230_dir / "ersp_keep.npy"
    f_npz = run230_dir / "ersp_keep.npz"

    if f_npy.exists():
        ersp = np.load(f_npy, allow_pickle=False)
        if ersp.ndim != 3:
            raise ValueError(f"Expected ersp_keep.npy shape (n,nF,nT), got {ersp.shape}")
    elif f_npz.exists():
        z = np.load(f_npz, allow_pickle=True)
        keys = sorted(z.files)
        ersp = np.stack([z[k].astype(np.float32, copy=False) for k in keys], axis=0)
    else:
        raise FileNotFoundError(f"Missing ERSPs: {f_npy} or {f_npz}")

    n, nF, nT = int(ersp.shape[0]), int(ersp.shape[1]), int(ersp.shape[2])

    # -------------------------
    # Pick sample-level meta parquet that matches ERSP rows
    # -------------------------
    candidates = [
        run230_dir / "df_keep_with_clusters.parquet",
        run230_dir / "df_keep.parquet",
        run230_dir / "df_meta.parquet",
    ]
    meta_path = None
    for p in candidates:
        if p.exists():
            try:
                df_try = pd.read_parquet(p)
                if len(df_try) == n:
                    meta_path = p
                    break
            except Exception:
                pass
    if meta_path is None:
        sizes = []
        for p in candidates:
            if p.exists():
                try:
                    sizes.append((p.name, len(pd.read_parquet(p))))
                except Exception:
                    sizes.append((p.name, "unreadable"))
        raise ValueError(
            "Could not find a meta parquet in 230 that matches ERSP rows.\n"
            f"ERSP rows: {n}\n"
            f"Tried: {sizes}\n"
            "Fix: ensure the parquet you want (df_keep_with_clusters.parquet or df_keep.parquet) "
            "has exactly the same sample count/order as ersp_keep.npy."
        )

    df = pd.read_parquet(meta_path).copy()
    df.columns = [c.strip() for c in df.columns]

    # Required identifiers
    for col in ["patient_id", "electrode"]:
        if col not in df.columns:
            raise ValueError(f"{meta_path.name} missing required column: {col}")
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["electrode"]  = df["electrode"].astype(str).str.strip()

    # Condition normalization (optional)
    cond_col = None
    for cand in ["condition", "conditions"]:
        if cand in df.columns:
            cond_col = cand
            break
    if cond_col is not None:
        s = df[cond_col].astype(str).fillna("NA")
        df["condition"] = s.apply(lambda x: x.split(",")[0].strip() if "," in x else x.strip())
        df.loc[df["condition"].isin(["", "nan", "None"]), "condition"] = "NA"
    else:
        df["condition"] = "NA"

    # -------------------------
    # Validate window bins (INCLUSIVE)
    # -------------------------
    f0, f1 = int(f_bins[0]), int(f_bins[1])
    t0, t1 = int(t_bins[0]), int(t_bins[1])
    if f0 > f1: f0, f1 = f1, f0
    if t0 > t1: t0, t1 = t1, t0

    if f0 < 0 or t0 < 0 or f1 >= nF or t1 >= nT:
        raise ValueError(
            f"Window out of bounds.\n"
            f"  f_bins={f_bins} valid: [0..{nF-1}]\n"
            f"  t_bins={t_bins} valid: [0..{nT-1}]"
        )

    # -------------------------
    # Sample filter by condition (keep ERSP aligned)
    # -------------------------
    mask = np.ones(n, dtype=bool)
    if condition is not None:
        condition = str(condition).strip()
        mask &= (df["condition"] == condition)

    if mask.sum() == 0:
        raise ValueError(f"No samples left after condition filter: condition={condition}")

    df_s = df.loc[mask, ["patient_id", "electrode", "condition"]].copy()
    ersp_s = ersp[mask]

    # -------------------------
    # Compute per-sample scalar from ERSP window
    # -------------------------
    win = ersp_s[:, f0:f1+1, t0:t1+1]
    sample_val = np.nanmean(win, axis=(1, 2)).astype(np.float32)

    df_s["sample_val"] = sample_val
    df_s = df_s.replace([np.inf, -np.inf], np.nan).dropna(subset=["sample_val"]).copy()
    if df_s.empty:
        raise ValueError("All sample values are NaN/Inf after window averaging.")

    # -------------------------
    # Aggregate to contact-level scalar
    # -------------------------
    agg_contact = str(agg_contact).strip().lower()
    if agg_contact == "mean":
        df_c = df_s.groupby(["patient_id", "electrode"], as_index=False)["sample_val"].mean()
    elif agg_contact == "median":
        df_c = df_s.groupby(["patient_id", "electrode"], as_index=False)["sample_val"].median()
    elif agg_contact == "maxabs":
        df_c = df_s.groupby(["patient_id", "electrode"], as_index=False)["sample_val"].apply(
            lambda x: float(np.max(np.abs(np.asarray(x, dtype=float))))
        ).reset_index()
        df_c = df_c.rename(columns={"sample_val": "sample_val"})
    else:
        raise ValueError("agg_contact must be one of: mean | median | maxabs")

    # -------------------------
    # Merge contact scalars with fsaverage coords
    # -------------------------
    df_coords = pd.read_csv(coords_path, sep="\t").copy()
    df_coords.columns = [c.strip() for c in df_coords.columns]
    if "name" in df_coords.columns and "electrode" not in df_coords.columns:
        df_coords = df_coords.rename(columns={"name": "electrode"})

    for col in ["patient_id", "electrode"]:
        df_coords[col] = df_coords[col].astype(str).str.strip()
    for col in ["x", "y", "z"]:
        df_coords[col] = pd.to_numeric(df_coords[col], errors="coerce")

    df_m = df_c.merge(df_coords[["patient_id","electrode","x","y","z"]].dropna(), on=["patient_id","electrode"], how="inner")
    if df_m.empty:
        raise ValueError("No contacts matched between contact-level values and coords_out. Check electrode naming consistency.")

    # -------------------------
    # Load fsaverage pial vertices (for distance + rendering)
    # -------------------------
    # NOTE: surfaces are in fsaverage tkrRAS, consistent with your coords_out.
    fsavg_dir = Path(__import__("functions.lf_blob_recon_config", fromlist=["FSAVERAGE_DIR"]).FSAVERAGE_DIR)
    lh_v, lh_f = read_geometry(str(fsavg_dir / "surf" / "lh.pial"))
    rh_v, rh_f = read_geometry(str(fsavg_dir / "surf" / "rh.pial"))
    lh_tree = KDTree(lh_v)
    rh_tree = KDTree(rh_v)

    pts = df_m[["x","y","z"]].to_numpy(dtype=float)
    d_lh, _ = lh_tree.query(pts, k=1)
    d_rh, _ = rh_tree.query(pts, k=1)
    d_min = np.minimum(d_lh, d_rh).astype(np.float32)
    df_m["dist_to_pial_mm"] = d_min

    if exclude_contacts_dist_to_pial_mm_gt is not None:
        thr = float(exclude_contacts_dist_to_pial_mm_gt)
        before = len(df_m)
        df_m = df_m.loc[df_m["dist_to_pial_mm"] <= thr].copy()
        if df_m.empty:
            raise ValueError(f"All contacts excluded by dist_to_pial_mm <= {thr}. Consider increasing threshold.")
        # print(f"[INFO] Kept {len(df_m)}/{before} contacts after pial distance filter (thr={thr}mm)")

    # -------------------------
    # Build pial meshes for rendering
    # -------------------------
    def _to_pv(v, f):
        faces = np.hstack([np.full((f.shape[0], 1), 3, dtype=np.int64), f]).ravel()
        m = pv.PolyData(v, faces)
        m.compute_normals(inplace=True)
        return m

    lh_mesh = _to_pv(lh_v, lh_f)
    rh_mesh = _to_pv(rh_v, rh_f)

    # -------------------------
    # Project contact values to each vertex via kNN + Gaussian weights
    # -------------------------
    c_xyz = df_m[["x","y","z"]].to_numpy(dtype=float)
    c_val = df_m["sample_val"].to_numpy(dtype=float)

    c_tree = KDTree(c_xyz)

    def _project(vertices):
        d, idx = c_tree.query(vertices, k=min(int(k_nearest), len(c_xyz)))
        if d.ndim == 1:
            d = d[:, None]
            idx = idx[:, None]
        w = np.exp(-(d ** 2) / (2.0 * float(sigma_mm) ** 2)).astype(np.float32)
        wsum = np.sum(w, axis=1)
        num = np.sum(w * c_val[idx], axis=1)
        vtx_val = (num / np.maximum(wsum, 1e-8)).astype(np.float32)
        return vtx_val, wsum.astype(np.float32)

    lh_val, lh_den = _project(lh_v)
    rh_val, rh_den = _project(rh_v)

    # density normalization for “darken low-density”
    def _norm_density(den):
        q = np.quantile(den, float(density_norm_q)) if len(den) else 1.0
        q = max(float(q), 1e-8)
        x = np.clip(den / q, 0.0, 1.0)
        return x.astype(np.float32)

    lh_den_n = _norm_density(lh_den)
    rh_den_n = _norm_density(rh_den)

    # -------------------------
    # Convert activity scalars to RGBA and apply density modulation
    # -------------------------
    cmap = mpl.cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)

    # --- ADD these new kwargs to your function signature (keep everything else the same) ---
    # density_alpha: bool = True,
    # alpha_min: float = 0.10,
    # alpha_max: float = 1.00,
    # alpha_gamma: float = 0.70,
    # base_brain_gray_rgb: tuple = (0.22, 0.22, 0.22),

    # --- REPLACE your _rgba_from_vals with this (keep the rest unchanged) ---
    def _rgba_from_vals(vals, den_n):
        rgba = cmap(norm(vals))  # float [0,1], shape (N,4)

        # Option A (recommended): use density -> ALPHA (gray base shows through)
        if "density_alpha" in locals() and density_alpha:
            a = float(alpha_min) + (float(alpha_max) - float(alpha_min)) * (den_n ** float(alpha_gamma))
            rgba[:, 3] = np.clip(a, 0.0, 1.0)

        # Optional legacy behavior: density -> RGB brightness (can keep off by setting s_min=s_max=1)
        if density_saturation:
            s = float(s_min) + (float(s_max) - float(s_min)) * (den_n ** float(density_gamma))
            s = np.clip(s, 0.0, 2.0)
            rgba[:, :3] = np.clip(rgba[:, :3] * s[:, None], 0.0, 1.0)

        return (rgba * 255).astype(np.uint8)


    lh_rgba = _rgba_from_vals(lh_val, lh_den_n)
    rh_rgba = _rgba_from_vals(rh_val, rh_den_n)

    # -------------------------
    # Output paths
    # -------------------------
    if views is None:
        views = ["left", "right", "dorsal", "ventral"]

    cond_tag = "all" if condition is None else str(condition)
    tag = (
        f"act_{cond_tag}"
        f"__f{f0:03d}-{f1:03d}"
        f"__t{t0:03d}-{t1:03d}"
        f"__agg-{agg_contact}"
        f"__sig{float(sigma_mm):g}"
        f"__k{int(k_nearest)}"
    )

    out_dir = Path(P["reports_out"]) / out_subdir / cond_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # cache arrays (useful for QA / later reuse)
    np.savez_compressed(
        Path(P["atlas_cache"]) / f"{tag}.npz",
        lh_val=lh_val, rh_val=rh_val,
        lh_den=lh_den, rh_den=rh_den,
        lh_den_n=lh_den_n, rh_den_n=rh_den_n,
        meta=dict(
            run_id_230=str(run_id_230),
            run_id_recon=str(run_id_recon),
            meta_path=str(meta_path),
            coords_path=str(coords_path),
            condition=str(condition),
            f_bins=(f0,f1), t_bins=(t0,t1),
            agg_contact=str(agg_contact),
            sigma_mm=float(sigma_mm),
            k_nearest=int(k_nearest),
            vmin=float(vmin), vmax=float(vmax),
            exclude_contacts_dist_to_pial_mm_gt=float(exclude_contacts_dist_to_pial_mm_gt),
        )
    )

    # -------------------------
    # Render views
    # -------------------------
    pv.set_jupyter_backend("static")
    try:
        pv.global_theme.multi_samples = 0
    except Exception:
        pass

    bounds = (
        min(lh_mesh.bounds[0], rh_mesh.bounds[0]), max(lh_mesh.bounds[1], rh_mesh.bounds[1]),
        min(lh_mesh.bounds[2], rh_mesh.bounds[2]), max(lh_mesh.bounds[3], rh_mesh.bounds[3]),
        min(lh_mesh.bounds[4], rh_mesh.bounds[4]), max(lh_mesh.bounds[5], rh_mesh.bounds[5]),
    )
    cams = compute_cameras(bounds)
    
    # --- REPLACE your rendering block inside the for view in views loop with this ---
    for view in views:
        pl = pv.Plotter(off_screen=True, window_size=(1200, 1000))

        # 1) base gray brain (always opaque)
        pl.add_mesh(lh_mesh, color=base_brain_gray_rgb, opacity=1.0, smooth_shading=True)
        pl.add_mesh(rh_mesh, color=base_brain_gray_rgb, opacity=1.0, smooth_shading=True)

        # 2) overlay activity with per-vertex RGBA (alpha encodes density)
        m1 = lh_mesh.copy(deep=True); m1.point_data["rgba"] = lh_rgba
        m2 = rh_mesh.copy(deep=True); m2.point_data["rgba"] = rh_rgba
        pl.add_mesh(m1, scalars="rgba", rgba=True, opacity=1.0, smooth_shading=True)
        pl.add_mesh(m2, scalars="rgba", rgba=True, opacity=1.0, smooth_shading=True)

        pl.camera_position = cams[view]
        pl.reset_camera_clipping_range()

        out_png = activity_frame_path_241(
            run_id_230, run_id_recon,
            condition=condition,
            view=view,
            f_bins=(f0, f1),
            t_bins=(t0, t1),
        )
        pl.screenshot(str(out_png), transparent_background=True, scale=2)
        pl.close()

    return out_dir

# ============================
# 241: Atlas activity outputs
# ============================
# Drop this block into lf_blob_recon.py (near other path/IO helpers).
# Python 3.6+ compatible.

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict

import functions.lf_blob_recon_config as C


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run241_dir(run_id_230: str, run_id_recon: Optional[str] = None) -> Path:
    """
    outputs/241_atlas_activity_plots/run230_<RUN_ID_230>__recon_<RUN_ID_RECON>/
    """
    rid230 = str(run_id_230).strip()
    ridrec = str(run_id_recon).strip() if run_id_recon else rid230

    # If you prefer, add these constants to config; otherwise we default safely.
    rootname = getattr(C, "RUN241_ROOTNAME", "241_atlas_activity_plots")
    run241_root = C.OUTPUTS_ROOT / rootname

    return run241_root / ("run230_%s__recon_%s" % (rid230, ridrec))


def ensure_dirs_241(run_id_230: str, run_id_recon: Optional[str] = None) -> Dict[str, Path]:
    run241 = _run241_dir(run_id_230, run_id_recon)

    frames_dirname = getattr(C, "RUN241_FRAMES_DIR", "frames")
    videos_dirname = getattr(C, "RUN241_VIDEOS_DIR", "videos")
    manifest_name = getattr(C, "RUN241_MANIFEST_NAME", "manifest.json")
    index_name = getattr(C, "RUN241_FRAMES_INDEX_NAME", "frames_index.tsv")

    frames = run241 / frames_dirname
    videos = run241 / videos_dirname
    run241.mkdir(parents=True, exist_ok=True)
    frames.mkdir(parents=True, exist_ok=True)
    videos.mkdir(parents=True, exist_ok=True)

    return {
        "run241_dir": run241,
        "frames_root": frames,
        "videos_root": videos,
        "manifest": run241 / manifest_name,
        "frames_index": run241 / index_name,
    }


def _read_json_safe(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_manifest_241(manifest_path: Path, payload: dict) -> None:
    """
    Idempotent manifest writer:
      - preserves existing keys unless overwritten by payload
      - writes created_utc once
      - updates updated_utc every call
    """
    old = _read_json_safe(manifest_path)

    out = dict(old)
    out.update(payload)

    if "schema_version" not in out:
        out["schema_version"] = 1
    if "created_utc" not in out:
        out["created_utc"] = _utc_now_iso()
    out["updated_utc"] = _utc_now_iso()

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")


def _append_frame_index_row(frames_index_path: Path, row: dict) -> None:
    """
    Appends a TSV row for each generated frame (lightweight provenance).
    Creates file with header if missing.
    """
    frames_index_path.parent.mkdir(parents=True, exist_ok=True)

    # Stable column order
    cols = [
        "created_utc",
        "run_id_230",
        "run_id_recon",
        "condition",
        "view",
        "f0",
        "f1",
        "t0",
        "t1",
        "png_path",
    ]
    row_out = {c: "" for c in cols}
    row_out.update(row)
    row_out["created_utc"] = row_out.get("created_utc") or _utc_now_iso()

    header = "\t".join(cols) + "\n"
    line = "\t".join(str(row_out[c]) for c in cols) + "\n"

    if not frames_index_path.exists():
        frames_index_path.write_text(header, encoding="utf-8")
    with frames_index_path.open("a", encoding="utf-8") as f:
        f.write(line)


def activity_frame_path_241(
    run_id_230: str,
    run_id_recon: str,
    *,
    condition: Optional[str],
    view: str,
    f_bins: Tuple[int, int],
    t_bins: Tuple[int, int],
    writer: str = "render_surface_activity_from_ersp_window",
    write_index_row: bool = False,
) -> Path:
    """
    Returns the PNG path for one frame under the required layout:

      241_atlas_activity_plots/
        run230_<RUN_ID_230>__recon_<RUN_ID_RECON>/
          manifest.json
          frames/
            cond-picture/
              view-left/
                f000-002/
                  t000-004.png

    If write_index_row=True, it also appends a row to frames_index.tsv.
    """
    rid230 = str(run_id_230).strip()
    ridrec = str(run_id_recon).strip()
    P241 = ensure_dirs_241(rid230, ridrec)

    # Write/update manifest (idempotent)
    _write_manifest_241(
        P241["manifest"],
        {
            "run_id_230": rid230,
            "run_id_recon": ridrec,
            "writer": writer,
            "outputs_root": str(C.OUTPUTS_ROOT),
            "run241_dir": str(P241["run241_dir"]),
            "frames_root": str(P241["frames_root"]),
            "videos_root": str(P241["videos_root"]),
            "frames_index": str(P241["frames_index"]),
        },
    )

    cond = (str(condition).strip().lower() if condition else "all")
    v = str(view).strip().lower()

    f0, f1 = int(f_bins[0]), int(f_bins[1])
    t0, t1 = int(t_bins[0]), int(t_bins[1])

    # normalize ordering
    if f0 > f1:
        f0, f1 = f1, f0
    if t0 > t1:
        t0, t1 = t1, t0

    ftag = "f%03d-%03d" % (f0, f1)
    tname = "t%03d-%03d.png" % (t0, t1)

    out_dir = P241["frames_root"] / ("cond-%s" % cond) / ("view-%s" % v) / ftag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / tname

    if write_index_row:
        _append_frame_index_row(
            P241["frames_index"],
            {
                "run_id_230": rid230,
                "run_id_recon": ridrec,
                "condition": cond,
                "view": v,
                "f0": f0,
                "f1": f1,
                "t0": t0,
                "t1": t1,
                "png_path": str(out_png),
            },
        )

    return out_png
