"""
build_fsaverage_meshes.py — one-time prep for the MOBA 3D brain viewer.

Reads FreeSurfer's fsaverage `lh.pial` and `rh.pial` surface files, converts
them to Niivue's compact .mz3 format, and writes them to:

    02_FBM_Clustering/outputs/250_recon/fsaverage/meshes/
        fsaverage_lh.mz3
        fsaverage_rh.mz3

Why .mz3: ~5–10x smaller than .gii, single-file per hemisphere, and Niivue
loads it natively in the browser. We also write .gii alongside as a fallback
(some Niivue versions handle .gii better; either format will work in the page).

Usage on the server:

    cd S:\\HumanNeuronLab\\ANALYSIS\\FLM\\Analysis_LoraFanda
    python 02_FBM_Clustering/scripts/build_fsaverage_meshes.py

If your fsaverage subjects directory is somewhere unusual, edit the
`FSAVERAGE_DIR` constant below before running.

Dependencies: nibabel (install with `pip install nibabel` if missing).

Once the .mz3 files exist, commit them to the repo:

    git add 02_FBM_Clustering/outputs/250_recon/fsaverage/meshes
    git commit -m "fsaverage meshes for MOBA 3D brain viewer"
    git push

Tip: run with your analysis conda env's Python (the one that has nibabel).
On Lora's machine that's the same Python you use for the clustering notebooks
(NOT C:\\...\\Python36\\python.exe — that one's too old and likely missing
nibabel). Activate the env first, e.g. `conda activate LORA_FLM_env1`.
"""

import gzip
import os
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel is required. Install with `pip install nibabel`.")
    sys.exit(1)

# --------------------------------------------------------------------------
# Paths — edit if your FreeSurfer fsaverage lives somewhere else
# --------------------------------------------------------------------------
NASAC = r"\\nasac-m2.unige.ch\m-HumanNeuronLab"

# Try the standard places where Lora's fsaverage might live; first hit wins.
FSAVERAGE_CANDIDATES = [
    Path(NASAC) / "DATARAW" / "SEEG_EXPERIMENTS_BERN" / "Reconstruction" / "fsaverage" / "surf",
    # Common FreeSurfer install path on Windows under conda/MNE-Python
    Path(os.environ.get("FREESURFER_HOME", "")) / "subjects" / "fsaverage" / "surf",
    Path(os.environ.get("SUBJECTS_DIR", "")) / "fsaverage" / "surf",
]

# Output destination
OUT_DIR = Path("02_FBM_Clustering/outputs/250_recon/fsaverage/meshes")


# --------------------------------------------------------------------------
# .mz3 writer (compact format used natively by Niivue)
# --------------------------------------------------------------------------
# Format reference: https://github.com/neurolabusc/MZ3
# Header (gzipped): magic 0x6d23 (uint16), attr (uint16, 1 = vertices+faces),
# nface (uint32), nvert (uint32), nskip (uint32, 0).
# Body: faces (nface × 3 × uint32), vertices (nvert × 3 × float32).
def write_mz3(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"verts must be (N, 3); got {verts.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (M, 3); got {faces.shape}")

    n_face = int(faces.shape[0])
    n_vert = int(verts.shape[0])
    header = struct.pack(
        "<HHIII",
        0x6D23,    # magic
        1,         # attr: vertices + faces present
        n_face,
        n_vert,
        0,         # nskip
    )
    body = faces.tobytes() + verts.tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(header)
        f.write(body)


# --------------------------------------------------------------------------
# .gii writer (fallback, native nibabel)
# --------------------------------------------------------------------------
def write_gii(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    g = nib.gifti.GiftiImage()
    g.add_gifti_data_array(nib.gifti.GiftiDataArray(verts, intent="NIFTI_INTENT_POINTSET"))
    g.add_gifti_data_array(nib.gifti.GiftiDataArray(faces, intent="NIFTI_INTENT_TRIANGLE"))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(g, str(path))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def find_fsaverage_surf_dir() -> Path:
    for cand in FSAVERAGE_CANDIDATES:
        if cand and cand.exists() and (cand / "lh.pial").exists():
            return cand
    raise FileNotFoundError(
        "Could not locate fsaverage/surf with lh.pial. Tried:\n  "
        + "\n  ".join(str(c) for c in FSAVERAGE_CANDIDATES)
        + "\n\nEdit FSAVERAGE_CANDIDATES in this script to point at your fsaverage."
    )


def export_surface(surf_dir: Path, hemi: str, kind: str) -> None:
    """
    kind = "pial" -> reads {hemi}.pial   -> fsaverage_{hemi}.mz3
    kind = "inflated" -> reads {hemi}.inflated -> fsaverage_{hemi}.inflated.mz3
    """
    src = surf_dir / f"{hemi}.{kind}"
    if not src.exists():
        print(f"  [skip] {src} not found")
        return
    verts, faces = nib.freesurfer.read_geometry(str(src))
    print(f"  {hemi}.{kind:<10s} : {verts.shape[0]:>7d} verts, {faces.shape[0]:>7d} faces")

    name_kind = "" if kind == "pial" else f".{kind}"
    mz3_path = OUT_DIR / f"fsaverage_{hemi}{name_kind}.mz3"
    gii_path = OUT_DIR / f"fsaverage_{hemi}{name_kind}.gii"

    write_mz3(mz3_path, verts, faces)
    write_gii(gii_path, verts, faces)

    print(f"    -> {mz3_path.name}  ({mz3_path.stat().st_size/1024:.0f} KB)")
    print(f"    -> {gii_path.name}  ({gii_path.stat().st_size/1024:.0f} KB)")


def export_volume_t1(fsaverage_mri_dir: Path) -> None:
    """Convert fsaverage T1.mgz -> fsaverage_T1.nii.gz for MOBA's volume layer."""
    mgz = fsaverage_mri_dir / "T1.mgz"
    if not mgz.exists():
        print(f"  [skip] {mgz} not found — volume overlay will be unavailable in MOBA")
        return
    img = nib.load(str(mgz))
    out = OUT_DIR / "fsaverage_T1.nii.gz"
    nib.save(img, str(out))
    sz_mb = out.stat().st_size / (1024*1024)
    print(f"  T1.mgz -> {out.name}  ({sz_mb:.1f} MB)")
    if sz_mb > 25:
        print(f"  [warn] {out.name} is large; consider mri_convert with -odt uchar to shrink.")


def main() -> None:
    surf_dir = find_fsaverage_surf_dir()
    fsaverage_root = surf_dir.parent  # parent of surf/
    print(f"[fsaverage] Reading from {fsaverage_root}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Surfaces: pial (existing behaviour) + inflated (new, for MOBA inflated toggle)
    for hemi in ("lh", "rh"):
        export_surface(surf_dir, hemi, "pial")
        export_surface(surf_dir, hemi, "inflated")

    # Volume: T1 for the MOBA volume overlay
    export_volume_t1(fsaverage_root / "mri")

    print("\nDone. Commit the new mesh + volume files:")
    print("  git add 02_FBM_Clustering/outputs/250_recon/fsaverage/meshes")
    print("  git commit -m 'fsaverage meshes + T1 for MOBA 3D brain viewer'")
    print("  git push")


if __name__ == "__main__":
    main()
