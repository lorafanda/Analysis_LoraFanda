# lf_blob_recon.py
"""
fsaverage coverage visualization (no clustering, no CT fiddling).

This module:

  1. Loads electrode coordinates from collaborator bundles:

         \\nasac-m2.unige.ch\m-HumanNeuronLab\#SHARE\To_send_collaborators\PAT_XXXX\
             BIDS/ieeg/sub-XXXX_electrodes.tsv   (primary)
             BIDS/sub-XXXX_electrodes.tsv        (fallback)

  2. Concatenates coverage across patients in a simple table:
         patient_id, electrode, x, y, z, isSubdural, hemisphere, source_space

  3. For now, assumes these coordinates are already in a common MRI RAS
     space that you are comfortable treating as "fsaverage-like" for
     high-level coverage visualization (no extra registration).

  4. Renders:
       - fsaverage + colored electrodes using PyVista (per-patient colors)
       - optional MNE-Python alignment snapshot
       - optional simple coverage STC on fsaverage (nearest-vertex counts)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import pyvista as pv
from nibabel.freesurfer.io import read_geometry, read_annot

import functions.lf_blob_recon_config as C


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------


def log(msg: str) -> None:
    """Simple stdout logger."""
    print(msg, flush=True)


# ---------------------------------------------------------------------
# Path helpers (PAT_XXXX bundles)
# ---------------------------------------------------------------------


def patient_freesurfer_dir(pid: str) -> Path:
    """
    Resolve the per-patient directory for collaborator bundles.

    Expected layout:
        <SHARE_PAT_ROOT>/
            PAT_XXXX/
                BIDS/
                elec_recon/
                label/
                mri/
                surf/
    """
    pid = str(pid).strip()
    if not pid.upper().startswith("PAT_"):
        raise ValueError(
            f"patient_freesurfer_dir() expects PAT_XXXX ids under SHARE_PAT_ROOT, got {pid!r}"
        )
    subj_dir = C.SHARE_PAT_ROOT / pid
    if not subj_dir.exists():
        raise FileNotFoundError(f"{pid}: subject directory not found: {subj_dir}")
    return subj_dir


def _bids_electrodes_path(pid: str) -> Path:
    """
    Resolve the BIDS electrodes.tsv path in a PAT_XXXX collaborator package.

    Layout (preferred):
        PAT_XXXX/BIDS/ieeg/sub-XXXX_electrodes.tsv

    Fallback:
        PAT_XXXX/BIDS/sub-XXXX_electrodes.tsv
    """
    pid = str(pid).strip()
    if not pid.upper().startswith("PAT_"):
        raise ValueError(
            f"_bids_electrodes_path supports only PAT_XXXX ids, got {pid!r}"
        )

    num = re.sub(r"^(PAT_|pat_)", "", pid)
    fname = C.BIDS_ELECTRODES_TEMPLATE.format(num=num)

    root = C.SHARE_PAT_ROOT / pid
    cand1 = root / C.BIDS_ELECTRODES_SUBDIR / "ieeg" / fname
    cand2 = root / C.BIDS_ELECTRODES_SUBDIR / fname

    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2

    # Return primary candidate for nicer error messages
    return cand1


# ---------------------------------------------------------------------
# Electrode loading / coverage table (BIDS-based)
# ---------------------------------------------------------------------


def load_patient_contacts(pid: str) -> pd.DataFrame:
    """
    Load electrode contact coordinates for a single patient (coverage only).

    Primary source:
        BIDS/ieeg/sub-XXXX_electrodes.tsv inside <SHARE_PAT_ROOT>/PAT_XXXX/

    Optional override:
        C.CONTACTS_PATH_OVERRIDES[pid] can point to an absolute TSV/CSV.

    Returns
    -------
    df : pd.DataFrame
        Columns:
            patient_id
            electrode
            x, y, z        (float)
            isSubdural     (0/1, heuristic from BIDS "type" if present)
            hemisphere     ("L"/"R" if available, else None)
            source_space   (string, e.g. "mri")
    """
    pid = str(pid).strip()

    # 1) Optional manual override
    if pid in C.CONTACTS_PATH_OVERRIDES:
        p = Path(C.CONTACTS_PATH_OVERRIDES[pid])
        if not p.exists():
            raise FileNotFoundError(f"{pid}: override contacts path missing: {p}")
        sep = "\t" if p.suffix == ".tsv" else ","
        df = pd.read_csv(p, sep=sep)
        df.columns = [c.strip() for c in df.columns]

        if "name" in df.columns and "electrode" not in df.columns:
            df = df.rename(columns={"name": "electrode"})

        for c in ("x", "y", "z"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["patient_id"] = pid
        if "source_space" not in df.columns:
            df["source_space"] = C.BIDS_COORD_SPACE
        if "isSubdural" not in df.columns:
            df["isSubdural"] = 0
        if "hemisphere" not in df.columns:
            df["hemisphere"] = None

        cols = [
            "patient_id",
            "electrode",
            "x",
            "y",
            "z",
            "isSubdural",
            "hemisphere",
            "source_space",
        ]
        return df.dropna(subset=["x", "y", "z", "electrode"])[cols].copy()

    # 2) Default: BIDS electrodes.tsv
    bids_path = _bids_electrodes_path(pid)
    if not bids_path.exists():
        raise FileNotFoundError(f"{pid}: BIDS electrodes.tsv not found: {bids_path}")

    df = pd.read_csv(bids_path, sep="\t")
    df.columns = [c.strip() for c in df.columns]

    if "name" not in df.columns:
        raise ValueError(
            f"{bids_path}: missing 'name' column (expected a BIDS electrodes.tsv)"
        )

    df = df.rename(columns={"name": "electrode"})

    required = {"electrode", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{bids_path}: missing required columns {sorted(missing)}")

    out = pd.DataFrame(
        {
            "patient_id": pid,
            "electrode": df["electrode"].astype(str).str.strip(),
            "x": pd.to_numeric(df["x"], errors="coerce"),
            "y": pd.to_numeric(df["y"], errors="coerce"),
            "z": pd.to_numeric(df["z"], errors="coerce"),
        }
    )

    # Subdural vs depth heuristic from "type" if present
    if "type" in df.columns:
        t = df["type"].astype(str).str.lower()
        is_subd = t.isin({"grid", "strip", "surface", "subdural"})
        out["isSubdural"] = is_subd.astype(int)
    else:
        out["isSubdural"] = 0

    # Hemisphere if present
    if "hemisphere" in df.columns:
        hem = df["hemisphere"].astype(str).str.upper()
        hem = hem.where(hem.isin({"L", "R"}), other=None)
        out["hemisphere"] = hem
    else:
        out["hemisphere"] = None

    out["source_space"] = C.BIDS_COORD_SPACE

    return out.dropna(subset=["x", "y", "z", "electrode"]).copy()


def build_contacts_table(patient_ids: Sequence[str]) -> pd.DataFrame:
    """Concatenate contact tables across patients (coverage table)."""
    frames = [load_patient_contacts(pid) for pid in patient_ids]
    df = pd.concat(frames, ignore_index=True)
    return df


# ---------------------------------------------------------------------
# fsaverage surface loading (PyVista)
# ---------------------------------------------------------------------


def load_fsaverage_pial_pyvista() -> Tuple[pv.PolyData, pv.PolyData]:
    """
    Load fsaverage pial surfaces (lh, rh) as PyVista PolyData.
    """
    fs_dir = C.FSAVERAGE_DIR
    surf_dir = fs_dir / "surf"

    lh_path = surf_dir / "lh.pial"
    rh_path = surf_dir / "rh.pial"

    if not lh_path.exists() or not rh_path.exists():
        raise FileNotFoundError(f"fsaverage pial surfaces not found under {surf_dir}")

    lh_coords, lh_faces = read_geometry(str(lh_path))
    rh_coords, rh_faces = read_geometry(str(rh_path))

    def _to_pv(coords: np.ndarray, faces: np.ndarray) -> pv.PolyData:
        faces_pv = np.c_[np.full((faces.shape[0], 1), 3), faces].ravel()
        surf = pv.PolyData(coords, faces_pv)
        surf.compute_normals(inplace=True)
        return surf

    lh = _to_pv(lh_coords, lh_faces)
    rh = _to_pv(rh_coords, rh_faces)

    return lh, rh


# ---------------------------------------------------------------------
# Coordinate handling (identity for now)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# fsaverage parcellation loading (aparc / a2009s / DKTatlas)
# ---------------------------------------------------------------------


def load_fsaverage_parcellation(
    parcellation: str | None = None,
) -> dict:
    """
    Load fsaverage surface parcellation (.annot) for both hemispheres.

    Parameters
    ----------
    parcellation : str | None
        Name of the parcellation, e.g. "aparc", "aparc.a2009s", "aparc.DKTatlas".
        If None, uses C.FSAVERAGE_PARCELLATION_DEFAULT.

    Returns
    -------
    parc : dict
        {
          "name": <parcellation>,
          "lh": {
              "labels": np.ndarray (n_lh_vertices,),   # int ids per vertex
              "names": list[str],                     # FS region names
              "id_to_name": dict[int, str],
          },
          "rh": { ... same keys ... }
        }
    """
    if parcellation is None:
        parcellation = C.FSAVERAGE_PARCELLATION_DEFAULT

    label_dir = C.FSAVERAGE_LABEL_DIR

    hemi_data = {}
    for hemi in ("lh", "rh"):
        annot_path = label_dir / f"{hemi}.{parcellation}.annot"
        if not annot_path.exists():
            raise FileNotFoundError(f"fsaverage {parcellation}: missing {annot_path}")

        labels, ctab, names = read_annot(str(annot_path))

        # names is a numpy array of bytes → decode to str
        names_str = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in names]

        # Build mapping from integer label id to name
        # Note: in FS, 'labels' array entries correspond to row indices in ctab (0..n-1),
        # but we store a direct id->name mapping for convenience.
        id_to_name = {i: nm for i, nm in enumerate(names_str)}

        hemi_data[hemi] = {
            "labels": labels,
            "names": names_str,
            "id_to_name": id_to_name,
        }

    return {
        "name": parcellation,
        "lh": hemi_data["lh"],
        "rh": hemi_data["rh"],
    }


def native_to_fsaverage_identity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For now, do not perform any extra registration.

    We simply assume the BIDS coordinates are already in a space you
    consider acceptable for atlas-level coverage visualization.
    """
    return df.copy()

# ---------------------------------------------------------------------
# Parcel vertex colors from ROI groups
# ---------------------------------------------------------------------


def _hex_to_rgb01(hex_str: str) -> tuple[float, float, float]:
    """
    Convert '#rrggbb' -> (r,g,b) floats in [0,1].
    """
    h = hex_str.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected 6-hex color, got {hex_str!r}")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b)


def make_fsaverage_parcel_vertex_colors(
    lh: pv.PolyData,
    rh: pv.PolyData,
    parc: dict,
    roi_groups: dict[str, dict] | None = None,
    base_color: str = "#e0e0e0",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create per-vertex RGB colors for fsaverage lh/rh surfaces based on a
    parcellation + ROI groups.

    Parameters
    ----------
    lh, rh : pv.PolyData
        fsaverage pial surfaces.

    parc : dict
        Output of load_fsaverage_parcellation().

    roi_groups : dict or None
        See C.FSAVERAGE_ROI_GROUPS for structure:
            {
              "GroupName": {
                  "labels": [...],   # list of region names from .annot
                  "color": "#rrggbb",
              },
              ...
            }
        If None or empty, all vertices get base_color.

    base_color : str
        Hex color for all vertices not in any ROI group.

    Returns
    -------
    colors_lh, colors_rh : np.ndarray
        Arrays of shape (n_vertices, 3) with float RGB in [0,1].
    """
    n_lh = lh.n_points
    n_rh = rh.n_points

    labels_lh = parc["lh"]["labels"]
    labels_rh = parc["rh"]["labels"]
    id_to_name_lh = parc["lh"]["id_to_name"]
    id_to_name_rh = parc["rh"]["id_to_name"]

    if labels_lh.shape[0] != n_lh or labels_rh.shape[0] != n_rh:
        raise ValueError(
            f"Parcellation vertex count mismatch: lh {labels_lh.shape[0]} vs {n_lh}, "
            f"rh {labels_rh.shape[0]} vs {n_rh}"
        )

    # Default: everything in base color
    base_rgb = np.array(_hex_to_rgb01(base_color), dtype=float)
    colors_lh = np.tile(base_rgb, (n_lh, 1))
    colors_rh = np.tile(base_rgb, (n_rh, 1))

    if not roi_groups:
        return colors_lh, colors_rh

    # Build label->color mapping from roi_groups
    name_to_rgb: dict[str, np.ndarray] = {}
    for group_name, spec in roi_groups.items():
        labels = spec.get("labels", [])
        color_hex = spec.get("color", "#ff0000")
        rgb = np.array(_hex_to_rgb01(color_hex), dtype=float)
        for lab in labels:
            name_to_rgb[lab] = rgb

    # Apply mapping hemi-wise
    def _apply_colors(labels: np.ndarray, id_to_name: dict[int, str], colors: np.ndarray):
        for vid, lab_id in enumerate(labels):
            nm = id_to_name.get(int(lab_id))
            if nm in name_to_rgb:
                colors[vid, :] = name_to_rgb[nm]

    _apply_colors(labels_lh, id_to_name_lh, colors_lh)
    _apply_colors(labels_rh, id_to_name_rh, colors_rh)

    return colors_lh, colors_rh


# ---------------------------------------------------------------------
# PyVista-based fsaverage coverage rendering
# ---------------------------------------------------------------------


def _assign_patient_colors(patient_ids: Sequence[str]) -> Dict[str, str]:
    """Map patient_id -> color using C.PATIENT_COLOR_PALETTE."""
    palette = C.PATIENT_COLOR_PALETTE
    mapping: Dict[str, str] = {}
    ids = list(patient_ids)
    for i, pid in enumerate(ids):
        mapping[pid] = palette[i % len(palette)]
    return mapping


def render_coverage_fsaverage_pyvista(
    coverage_df: pd.DataFrame,
    out_dir: Path | str,
    views: Optional[Sequence[str]] = None,
    transparent_bg: bool = True,
) -> Dict[str, Path]:
    """
    Render coverage (electrodes) on fsaverage using PyVista.

    Parameters
    ----------
    coverage_df : DataFrame
        Must contain 'patient_id', 'x', 'y', 'z', 'isSubdural'.

    out_dir : Path or str
        Directory where one PNG per view will be saved.

    views : list of str | None
        If None, uses C.VIEWS_TO_SAVE.

    transparent_bg : bool
        If True, output PNGs have transparent background.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lh, rh = load_fsaverage_pial_pyvista()

    pv.global_theme.multi_samples = 0
    plotter = pv.Plotter(off_screen=True)

    if transparent_bg:
        plotter.set_background("white", top="white")
    else:
        plotter.set_background("white")

    # Brain surfaces
    plotter.add_mesh(
        lh,
        color=C.BRAIN_COLOR,
        opacity=C.BRAIN_OPACITY,
        specular=C.BRAIN_SPECULAR,
        specular_power=C.BRAIN_SPECULAR_POWER,
        ambient=C.BRAIN_AMBIENT,
        diffuse=C.BRAIN_DIFFUSE,
    )
    plotter.add_mesh(
        rh,
        color=C.BRAIN_COLOR,
        opacity=C.BRAIN_OPACITY,
        specular=C.BRAIN_SPECULAR,
        specular_power=C.BRAIN_SPECULAR_POWER,
        ambient=C.BRAIN_AMBIENT,
        diffuse=C.BRAIN_DIFFUSE,
    )

    # Patient colors
    pat_colors = _assign_patient_colors(sorted(coverage_df["patient_id"].unique()))

    # Electrodes
    for _, row in coverage_df.iterrows():
        x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
        pid = str(row["patient_id"])
        color = pat_colors.get(pid, "#000000")
        radius = (
            C.SUBDURAL_RADIUS
            if int(row.get("isSubdural", 0))
            else C.DEPTH_RADIUS
        )
        sphere = pv.Sphere(radius=radius, center=(x, y, z))
        plotter.add_mesh(
            sphere,
            color=color,
            opacity=C.ELECTRODE_OPACITY,
        )

    # Camera presets
    def _set_view(name: str) -> None:
        """
        Map our logical view names to PyVista camera presets.

        Supports both:
          - custom names: left, right, frontal, posterior, dorsal, ventral
          - raw PyVista codes: xy, xz, yz, yx, zx, zy, iso
        """
        # If user passes a raw PyVista code, forward it directly
        if name in {"xy", "xz", "yz", "yx", "zx", "zy", "iso"}:
            plotter.camera_position = name
            return

        # Otherwise interpret our semantic names
        if name == "left":
            plotter.camera_position = "yz"
        elif name == "right":
            plotter.camera_position = "-yz"
        elif name == "frontal":
            plotter.camera_position = "xz"
        elif name == "posterior":
            plotter.camera_position = "-xz"
        elif name == "dorsal":
            plotter.camera_position = "zx"
        elif name == "ventral":
            plotter.camera_position = "-zx"
        else:
            plotter.camera_position = "iso"

    if views is None:
        views = C.VIEWS_TO_SAVE

    out_paths: Dict[str, Path] = {}

    for v in views:
        _set_view(v)
        plotter.reset_camera()
        out_png = out_dir / f"fsaverage_coverage_{v}.png"
        plotter.show(
            screenshot=str(out_png),
            auto_close=False,
            window_size=(1600, 1200),
        )
        log(f"PyVista: wrote {out_png}")
        out_paths[v] = out_png

    plotter.close()
    return out_paths


# ---------------------------------------------------------------------
# fsaverage: parcels (aparc / a2009s / DKTatlas) + electrodes
# ---------------------------------------------------------------------


def render_fsaverage_parcels_with_coverage(
    coverage_df: pd.DataFrame | None,
    out_dir: Path | str,
    parcellation: str | None = None,
    roi_groups: dict[str, dict] | None = None,
    views: Optional[Sequence[str]] = None,
    transparent_bg: bool = True,
    with_electrodes: bool = True,
) -> Dict[str, Path]:
    """
    Render fsaverage with surface parcels highlighted (via .annot) and,
    optionally, electrode coverage on top.

    Parameters
    ----------
    coverage_df : DataFrame or None
        Same as for render_coverage_fsaverage_pyvista. If None and
        with_electrodes=True, no electrodes are drawn.

    out_dir : path-like
        Directory to save one PNG per view.

    parcellation : str | None
        Name of fsaverage parcellation ("aparc", "aparc.a2009s", "aparc.DKTatlas").
        If None, uses C.FSAVERAGE_PARCELLATION_DEFAULT.

    roi_groups : dict or None
        ROI group definition (see C.FSAVERAGE_ROI_GROUPS). If None, parcels
        are not grouped; all vertices are drawn in base_color.

    views : list of str | None
        View names. If None, uses C.VIEWS_TO_SAVE.

    transparent_bg : bool
        Transparent PNG background if True.

    with_electrodes : bool
        If True, electrodes from coverage_df are drawn on top of parcels.

    Returns
    -------
    out_paths : dict
        {view_name: Path_to_png}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lh, rh = load_fsaverage_pial_pyvista()
    parc = load_fsaverage_parcellation(parcellation)
    colors_lh, colors_rh = make_fsaverage_parcel_vertex_colors(
        lh, rh, parc, roi_groups=roi_groups, base_color="#e0e0e0"
    )

    # Attach vertex colors to meshes
    lh.point_data["parcel_rgb"] = colors_lh
    rh.point_data["parcel_rgb"] = colors_rh

    pv.global_theme.multi_samples = 0
    plotter = pv.Plotter(off_screen=True)

    if transparent_bg:
        plotter.set_background("white", top="white")
    else:
        plotter.set_background("white")

    # Brain surfaces with parcel colors
    plotter.add_mesh(
        lh,
        scalars="parcel_rgb",
        rgb=True,
        opacity=C.BRAIN_OPACITY,
        specular=C.BRAIN_SPECULAR,
        specular_power=C.BRAIN_SPECULAR_POWER,
        ambient=C.BRAIN_AMBIENT,
        diffuse=C.BRAIN_DIFFUSE,
    )
    plotter.add_mesh(
        rh,
        scalars="parcel_rgb",
        rgb=True,
        opacity=C.BRAIN_OPACITY,
        specular=C.BRAIN_SPECULAR,
        specular_power=C.BRAIN_SPECULAR_POWER,
        ambient=C.BRAIN_AMBIENT,
        diffuse=C.BRAIN_DIFFUSE,
    )

    # Optional electrodes
    if with_electrodes and coverage_df is not None and not coverage_df.empty:
        pat_colors = _assign_patient_colors(
            sorted(coverage_df["patient_id"].unique())
        )
        for _, row in coverage_df.iterrows():
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
            pid = str(row["patient_id"])
            color = pat_colors.get(pid, "#000000")
            radius = (
                C.SUBDURAL_RADIUS
                if int(row.get("isSubdural", 0))
                else C.DEPTH_RADIUS
            )
            sphere = pv.Sphere(radius=radius, center=(x, y, z))
            plotter.add_mesh(
                sphere,
                color=color,
                opacity=C.ELECTRODE_OPACITY,
            )

    # Camera presets – reuse the raw PyVista codes if given
    def _set_view(name: str) -> None:
        # Raw PyVista presets
        if name in {"xy", "xz", "yz", "yx", "zx", "zy", "iso"}:
            plotter.camera_position = name
            return

        # Semantic names
        if name == "left":
            plotter.camera_position = "yz"
        elif name == "right":
            plotter.camera_position = "-yz"
        elif name == "frontal":
            plotter.camera_position = "xz"
        elif name == "posterior":
            plotter.camera_position = "-xz"
        elif name == "dorsal":
            plotter.camera_position = "zx"
        elif name == "ventral":
            plotter.camera_position = "-zx"
        else:
            plotter.camera_position = "iso"

    if views is None:
        views = C.VIEWS_TO_SAVE

    out_paths: Dict[str, Path] = {}
    for v in views:
        _set_view(v)
        plotter.reset_camera()
        out_png = out_dir / f"fsaverage_parcels_{parc['name']}_{v}.png"
        plotter.show(
            screenshot=str(out_png),
            auto_close=False,
            window_size=(1600, 1200),
        )
        log(f"PyVista parcels: wrote {out_png}")
        out_paths[v] = out_png

    plotter.close()
    return out_paths



# ---------------------------------------------------------------------
# MNE-Python-based fsaverage visualization (optional)
# ---------------------------------------------------------------------


def make_mne_info_from_contacts(
    df: pd.DataFrame,
    sfreq: float = 1.0,
):
    """
    Create an MNE Info + DigMontage describing intracranial contacts
    in MRI coordinates (coverage only).

    Parameters
    ----------
    df : DataFrame
        Must contain 'electrode', 'x', 'y', 'z' in mm.

    sfreq : float
        Dummy sampling frequency.
    """
    import mne

    ch_names = df["electrode"].astype(str).tolist()
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="seeg")

    # mm -> m
    pos_mm = df[["x", "y", "z"]].to_numpy(dtype=float)
    pos_m = pos_mm / 1000.0

    ch_pos = {name: xyz for name, xyz in zip(ch_names, pos_m, strict=False)}

    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        coord_frame="mri",
    )
    info.set_montage(montage)
    return info, montage


def render_coverage_fsaverage_mne(
    coverage_df: pd.DataFrame,
    out_png: Path | str,
    sensor_color: Tuple[float, float, float, float] | None = None,
):
    """
    Render coverage using MNE-Python plot_alignment + snapshot_brain_montage.
    """
    import mne
    from mne.viz import plot_alignment, snapshot_brain_montage, set_3d_view

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if sensor_color is None:
        sensor_color = C.MNE_DEFAULT_SENSOR_COLOR

    info, montage = make_mne_info_from_contacts(coverage_df)

    mne.viz.set_3d_backend(C.MNE_3D_BACKEND)

    fig = plot_alignment(
        info,
        trans=None,
        subject=C.MNE_FSAV_SUBJECT,
        subjects_dir=C.MNE_SUBJECTS_DIR,
        coord_frame="mri",
        surfaces=["pial"],
        meg=False,
        eeg=False,
        seeg=True,
        dig=True,
        sensor_colors=sensor_color,
        show_axes=False,
    )

    set_3d_view(fig, azimuth=0, elevation=70, distance="auto", focalpoint="auto")

    xy, im = snapshot_brain_montage(fig, info)
    im.save(str(out_png))
    log(f"MNE: wrote {out_png}")

    return {"png": out_png, "xy": xy}


# ---------------------------------------------------------------------
# Simple coverage STC (optional)
# ---------------------------------------------------------------------


def compute_coverage_stc_fsaverage(
    coverage_df: pd.DataFrame,
    surf: str | None = None,
    max_dist_mm: float | None = None,
):
    """
    Build a simple fsaverage SourceEstimate where each contact contributes +1
    to the nearest surface vertex (coverage density).
    """
    import mne
    from scipy.spatial import cKDTree

    if surf is None:
        surf = C.COVERAGE_SURF_NAME
    if max_dist_mm is None:
        max_dist_mm = C.COVERAGE_MAX_DIST_MM

    subjects_dir = str(C.MNE_SUBJECTS_DIR)
    lh_coords, lh_tris = mne.surface.read_surface(
        f"{subjects_dir}/{C.MNE_FSAV_SUBJECT}/surf/lh.{surf}"
    )
    rh_coords, rh_tris = mne.surface.read_surface(
        f"{subjects_dir}/{C.MNE_FSAV_SUBJECT}/surf/rh.{surf}"
    )

    lh_n = lh_coords.shape[0]
    all_coords = np.vstack([lh_coords, rh_coords])
    tree = cKDTree(all_coords)

    pos_mm = coverage_df[["x", "y", "z"]].to_numpy(dtype=float)
    pos_m = pos_mm / 1000.0

    dists, idx = tree.query(pos_m, k=1)
    mask = dists <= (max_dist_mm / 1000.0)
    idx_keep = idx[mask]

    coverage = np.zeros(all_coords.shape[0], dtype=float)
    for i in idx_keep:
        coverage[i] += 1.0

    lh_data = coverage[:lh_n]
    rh_data = coverage[lh_n:]

    lh_idx = np.where(lh_data > 0)[0]
    rh_idx = np.where(rh_data > 0)[0]

    lh_vals = lh_data[lh_idx][:, np.newaxis]
    rh_vals = rh_data[rh_idx][:, np.newaxis]

    stc = mne.SourceEstimate(
        data=np.vstack([lh_vals, rh_vals]),
        vertices=[lh_idx, rh_idx],
        tmin=0.0,
        tstep=1.0,
        subject=C.MNE_FSAV_SUBJECT,
    )
    return stc


# ---------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# High-level driver: fsaverage parcel-highlighted views
# ---------------------------------------------------------------------


def run_fsaverage_parcel_views(
    patient_ids: Optional[Sequence[str]] = None,
    output_root: Optional[Path | str] = None,
    parcellation: str | None = None,
    roi_groups: Optional[dict[str, dict]] = None,
    with_electrodes: bool = True,
):
    """
    High-level convenience function:

      1. Build coverage table across patients (using BIDS fsaverage coords).
      2. Render fsaverage parcel-highlighted surfaces with optional electrodes.

    Parameters
    ----------
    patient_ids : list of str | None
        If None, uses C.PATIENT_IDS_DEFAULT.

    output_root : path-like | None
        If None, uses C.COVERAGE_OUTPUT_ROOT / "fsaverage_parcels".

    parcellation : str | None
        fsaverage parcellation name ("aparc", "aparc.a2009s", "aparc.DKTatlas").

    roi_groups : dict | None
        ROI grouping; if None, uses C.FSAVERAGE_ROI_GROUPS.

    with_electrodes : bool
        If True, overlay electrodes; otherwise just parcels.

    Returns
    -------
    outputs : dict
        {"parcels": {view_name: png_path}}
    """
    if patient_ids is None:
        patient_ids = C.PATIENT_IDS_DEFAULT

    if output_root is None:
        output_root = C.COVERAGE_OUTPUT_ROOT / "fsaverage_parcels"
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if roi_groups is None:
        roi_groups = C.FSAVERAGE_ROI_GROUPS

    log(f"Building coverage table for parcel views, patients: {patient_ids}")
    contacts = build_contacts_table(patient_ids)
    fsav_contacts = native_to_fsaverage_identity(contacts)

    parcels_dir = output_root / "pyvista"
    parcel_pngs = render_fsaverage_parcels_with_coverage(
        fsav_contacts if with_electrodes else None,
        parcels_dir,
        parcellation=parcellation,
        roi_groups=roi_groups,
        views=C.VIEWS_TO_SAVE,
        transparent_bg=True,
        with_electrodes=with_electrodes,
    )

    return {"parcels": parcel_pngs}


def run_fsaverage_coverage(
    patient_ids: Optional[Sequence[str]] = None,
    output_root: Optional[Path | str] = None,
    do_mne: bool = True,
    do_stc: bool = False,
):
    """
    High-level convenience function:

      1. Build coverage table across patients (PAT_XXXX).
      2. (Currently) use coordinates as-is for fsaverage coverage.
      3. Render PyVista coverage PNGs.
      4. Optionally render MNE coverage snapshot.
      5. Optionally build a coverage STC (source-like map).
    """
    if patient_ids is None:
        patient_ids = C.PATIENT_IDS_DEFAULT

    if output_root is None:
        output_root = C.COVERAGE_OUTPUT_ROOT / "fsaverage_coverage"
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    log(f"Building coverage table for patients: {patient_ids}")
    contacts = build_contacts_table(patient_ids)

    fsav_contacts = native_to_fsaverage_identity(contacts)

    outputs: Dict[str, Dict[str, Path]] = {}

    # 3) PyVista coverage
    pyvista_dir = output_root / "pyvista"
    pyvista_pngs = render_coverage_fsaverage_pyvista(fsav_contacts, pyvista_dir)
    outputs["pyvista"] = pyvista_pngs

    # 4) MNE coverage snapshot
    if do_mne:
        mne_png = output_root / "mne_fsaverage_coverage.png"
        mne_out = render_coverage_fsaverage_mne(fsav_contacts, mne_png)
        outputs["mne"] = {"png": mne_out["png"]}

    # 5) Simple coverage STC
    if do_stc:
        stc = compute_coverage_stc_fsaverage(fsav_contacts)
        stc_path = output_root / "fsaverage_coverage-stc.h5"
        stc.save(str(stc_path))
        log(f"STC: wrote {stc_path}")
        outputs["stc"] = {"stc_path": stc_path}

    return outputs


if __name__ == "__main__":
    outs = run_fsaverage_coverage()
    log("Done fsaverage coverage.")
    for k, v in outs.items():
        log(f"{k}:")
        for kk, vv in v.items():
            log(f"  {kk}: {vv}")
