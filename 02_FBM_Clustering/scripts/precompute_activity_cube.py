"""
precompute_activity_cube.py — data prep for the Activity Visualizer page.

Builds a compact data bundle that the browser page (activity_visualizer.html)
uses to render ERSP activity on the fsaverage brain, with a play button that
sweeps a sliding time window and a multiselect of frequency bands to average.

THE IDEA
--------
The surface projection (vertex value = Gaussian-weighted mean of nearby contact
values) is a *linear* operator whose weights depend ONLY on geometry, never on
the activity values. So instead of pre-rendering one image per (band, window,
condition), we precompute ONCE:

  1. a per-contact x band x time "cube", and
  2. the vertex->contact projection weights (geometry only),

and let the browser do band-averaging + time-window averaging + projection +
density-alpha live. Averaging a band subset or a time window in the browser is
mathematically identical to projecting that exact window (exact for the "mean"
contact aggregation).

SOURCE  (all inside Analysis_LoraFanda — Analysis_Lora is never used)
  - Raw ungated ERSP:  04_FBM_Pooling/outputs/_dataset/pooling/_raw_ungated/
        X_3d.npy                       (n_samples, 129, 300) float32 raw ERSP (ungated)
        df_meta.parquet                (row-aligned: patient_id, electrode, condition)
        (built from 01_FBM_Analysis/outputs/04_ersp_LM_RAWONLY)
  - Contact coords:    02_FBM_Clustering/outputs/250_recon/fsaverage/coords/ALL_PATIENTS_contacts_fsaverage.csv
        (patient, name, hemi, x, y, z, dist_to_pial_mm, is_cortical, ...)
  - Meshes:            02_FBM_Clustering/outputs/250_recon/fsaverage/meshes/fsaverage_{lh,rh}.gii

OUTPUT  (written next to the meshes)
  02_FBM_Clustering/outputs/250_recon/fsaverage/activity_viz/
      manifest.json            # everything the page needs to interpret the binaries
      cube_f32.bin             # [n_cond, n_band, n_contact, n_time] float32 (NaN = contact absent in condition)
      proj_lh_idx_u16.bin      # [nvert_lh, K] uint16  global contact indices
      proj_lh_w_f32.bin        # [nvert_lh, K] float32 RAW gaussian weights (un-normalized)
      proj_rh_idx_u16.bin
      proj_rh_w_f32.bin
      contacts.json            # per-contact metadata (for optional hover/debug)

Run with the Python that has numpy + pandas + pyarrow (Python 3.11 here):
    "C:/Users/fanda/AppData/Local/Programs/Python/Python311/python.exe" \
        02_FBM_Clustering/scripts/precompute_activity_cube.py
"""

import gzip
import json
import re
import struct
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG  (everything lives in Analysis_LoraFanda)
# --------------------------------------------------------------------------
REPO = Path(r"S:\HumanNeuronLab\ANALYSIS\FLM\Analysis_LoraFanda")
OUTPUTS = REPO / "02_FBM_Clustering" / "outputs"

# Raw ungated ERSP dataset (the actual activity) + row-aligned sample meta.
DATASET_DIR = REPO / "04_FBM_Pooling" / "outputs" / "_dataset" / "pooling" / "_raw_ungated"
ERSP_NPY = DATASET_DIR / "X_3d.npy"            # (n, 129, 300) float32, already 3D
META_PARQUET = DATASET_DIR / "df_meta.parquet"

# All-patient fsaverage contact coordinates (HUG + EL cohorts).
COORDS_CSV = OUTPUTS / "250_recon" / "fsaverage" / "coords" / "ALL_PATIENTS_contacts_fsaverage.csv"

MESH_DIR = OUTPUTS / "250_recon" / "fsaverage" / "meshes"
OUT_DIR = OUTPUTS / "250_recon" / "fsaverage" / "activity_viz"

# ERSP axes. NOTE: the ERSP pipeline runs in TIME-NORMALIZED ("TN") mode
# (config.py: mode="TN", proportions=(baseline, stim, post)=(0.0, 0.50, 0.50)).
# So the 300 time bins are NOT seconds — they are two time-warped phases
# concatenated: the first 50% is the STIMULUS phase (onset->offset, warped to a
# common length) and the second 50% is the RESPONSE/post phase (warped). Freq is
# linear 0..fmax Hz over nF bins.
NF, NT = 129, 300
FMAX_HZ = 500.0
PROPORTIONS = (0.0, 0.50, 0.50)          # (baseline, stim, post) — must sum to 1
PHASE_NAMES = ("Stimulus", "Response")   # the two warped phases (post baseline)

# Frequency bands (Hz) — the standard 8 used throughout the project.
F_BANDS_HZ = [
    (0, 5), (5, 10), (10, 16), (16, 40),
    (40, 70), (70, 130), (130, 250), (250, 500),
]

# Conditions, in display order. "all" = pool every sample regardless of condition.
CONDITIONS = ["all", "picture", "audio", "reading"]

# Projection / rendering params
K_NEAREST = 4
SIGMA_MM = 4.0
EXCLUDE_DIST_TO_PIAL_MM_GT = 12.0
AGG_CONTACT = "mean"          # "mean" keeps the linear-projection identity exact
CORTICAL_ONLY = True          # keep only is_cortical contacts (drop white-matter)

# Color / alpha defaults (page can tweak live)
VMIN, VMAX = -6.0, 6.0
CMAP = "bwr"
DENSITY_NORM_Q = 0.98
ALPHA_MIN, ALPHA_MAX, ALPHA_GAMMA = 0.10, 1.00, 0.70
BASE_BRAIN_GRAY_RGB = [0.55, 0.55, 0.55]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def norm_el(s) -> str:
    """Normalize an electrode label for matching: drop non-alphanumerics, upper-case.
    Handles 'A_L10' (ERSP meta) == 'AL10' (coords) and case differences."""
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()


def hz_to_bin(hz: float, nF: int, fmax_hz: float) -> int:
    hz = max(0.0, min(float(hz), float(fmax_hz)))
    return int(round(hz / float(fmax_hz) * (nF - 1)))


def band_hz_to_bins(band_hz, nF, fmax_hz):
    f_lo, f_hi = sorted(band_hz)
    b0, b1 = hz_to_bin(f_lo, nF, fmax_hz), hz_to_bin(f_hi, nF, fmax_hz)
    return (min(b0, b1), max(b0, b1))


def read_mz3_vertices(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic, attr, nface, nvert, nskip = struct.unpack_from("<HHIII", raw, 0)
    off = 16 + nskip + nface * 3 * 4
    verts = np.frombuffer(raw, dtype="<f4", count=nvert * 3, offset=off).reshape(nvert, 3)
    return np.ascontiguousarray(verts, dtype=np.float32)


def read_gii_vertices(path: Path) -> np.ndarray:
    """Parse a GIfTI POINTSET data array (base64/gzip) without nibabel."""
    import base64
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    for da in root.iter("DataArray"):
        if da.get("Intent") != "NIFTI_INTENT_POINTSET":
            continue
        dims = int(da.get("Dim0"))
        enc = da.get("Encoding")
        order = da.get("ArrayIndexingOrder", "RowMajorOrder")
        dtype = "<f4" if "FLOAT32" in da.get("DataType", "") else "<f8"
        data_el = da.find("Data")
        rawb = base64.b64decode(data_el.text.strip())
        if enc == "GZipBase64Binary":
            import zlib
            rawb = zlib.decompress(rawb)   # GIfTI "GZip" is really zlib (78 9c)
        arr = np.frombuffer(rawb, dtype=dtype).astype(np.float32)
        arr = arr.reshape(dims, 3) if order == "RowMajorOrder" else arr.reshape(3, dims).T
        return np.ascontiguousarray(arr, dtype=np.float32)
    raise ValueError(f"No POINTSET in {path}")


def hemi_vertices(hemi: str) -> np.ndarray:
    """Vertices for the projection. The .mz3 and .gii were written from the same
    surface in the same vertex order, so we read the .mz3 (fast, reliable) even
    though the page displays the .gii."""
    m = MESH_DIR / f"fsaverage_{hemi}.mz3"
    g = MESH_DIR / f"fsaverage_{hemi}.gii"
    return read_mz3_vertices(m) if m.exists() else read_gii_vertices(g)


def knn_gaussian(vertices, contacts, k, sigma_mm):
    """Per vertex: k nearest contacts + RAW gaussian weights (numpy, chunked)."""
    nv, m = vertices.shape[0], contacts.shape[0]
    k = min(k, m)
    c2 = np.einsum("ij,ij->i", contacts, contacts)
    idx_out = np.empty((nv, k), dtype=np.int32)
    w_out = np.empty((nv, k), dtype=np.float32)
    two_sig2 = 2.0 * float(sigma_mm) ** 2
    chunk = 16384
    for s in range(0, nv, chunk):
        e = min(s + chunk, nv)
        V = vertices[s:e]
        d2 = np.einsum("ij,ij->i", V, V)[:, None] + c2[None, :] - 2.0 * (V @ contacts.T)
        np.maximum(d2, 0.0, out=d2)
        part = np.argpartition(d2, k - 1, axis=1)[:, :k] if k < m \
            else np.broadcast_to(np.arange(m), (e - s, m)).copy()
        rows = np.arange(e - s)[:, None]
        idx_out[s:e] = part
        w_out[s:e] = np.exp(-d2[rows, part] / two_sig2).astype(np.float32)
    return idx_out, w_out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    # ---- Raw ERSP stack + row-aligned meta -------------------------------
    X = np.load(ERSP_NPY, mmap_mode="r")              # (n, 129, 300) or (n, nF*nT)
    if X.ndim == 3:
        ersp = X
    else:
        assert X.shape[1] == NF * NT, f"X width {X.shape[1]} != {NF}*{NT}"
        ersp = X.reshape(X.shape[0], NF, NT)
    n = ersp.shape[0]
    assert ersp.shape[1:] == (NF, NT), f"ersp shape {ersp.shape} != (*,{NF},{NT})"
    print(f"[ersp] {ERSP_NPY.name} -> ersp {ersp.shape}")

    df = pd.read_parquet(META_PARQUET)
    assert len(df) == n, f"meta rows {len(df)} != ersp rows {n}"
    df = df.reset_index(drop=True)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["electrode"] = df["electrode"].astype(str).str.strip()
    df["condition"] = df["condition"].astype(str).str.strip()
    df["el_key"] = df["electrode"].map(norm_el)

    # ---- Frequency bands -------------------------------------------------
    band_bins = [band_hz_to_bins(b, NF, FMAX_HZ) for b in F_BANDS_HZ]
    n_band = len(band_bins)
    print("[bands] Hz -> bins:", list(zip(F_BANDS_HZ, band_bins)))

    bandval = np.empty((n, n_band, NT), dtype=np.float32)
    for bi, (b0, b1) in enumerate(band_bins):
        bandval[:, bi, :] = np.nanmean(ersp[:, b0:b1 + 1, :], axis=1)

    # ---- Contacts: ERSP samples joined to fsaverage coords ---------------
    co = pd.read_csv(COORDS_CSV)
    co.columns = [c.strip() for c in co.columns]
    co["patient_id"] = co["patient"].astype(str).str.strip()
    co["el_key"] = co["name"].map(norm_el)
    for c in ("x", "y", "z", "dist_to_pial_mm"):
        co[c] = pd.to_numeric(co[c], errors="coerce")
    co = co.drop_duplicates(subset=["patient_id", "el_key"])

    df_contacts = df[["patient_id", "electrode", "el_key"]].drop_duplicates().reset_index(drop=True)
    cols = ["patient_id", "el_key", "x", "y", "z", "dist_to_pial_mm", "hemi"]
    if "is_cortical" in co.columns:
        cols.append("is_cortical")
    merged = df_contacts.merge(co[cols], on=["patient_id", "el_key"], how="inner")
    before = len(merged)
    merged = merged.dropna(subset=["x", "y", "z"])
    if CORTICAL_ONLY and "is_cortical" in merged.columns:
        merged = merged[merged["is_cortical"].astype(float) == 1]
    merged = merged[merged["dist_to_pial_mm"] <= EXCLUDE_DIST_TO_PIAL_MM_GT].reset_index(drop=True)
    n_contact = len(merged)
    print(f"[contacts] {n_contact} kept (of {before} matched, {df_contacts.shape[0]} in ERSP) "
          f"cortical_only={CORTICAL_ONLY} dist<= {EXCLUDE_DIST_TO_PIAL_MM_GT}mm")

    merged["cidx"] = np.arange(n_contact, dtype=np.int64)
    key2cidx = {(p, e): i for p, e, i in
                zip(merged["patient_id"], merged["el_key"], merged["cidx"])}
    samp_cidx = np.array(
        [key2cidx.get((p, e), -1) for p, e in zip(df["patient_id"], df["el_key"])],
        dtype=np.int64,
    )

    # ---- Cube  [cond, band, contact, time]  (mean over samples) ----------
    n_cond = len(CONDITIONS)
    cube = np.full((n_cond, n_band, n_contact, NT), np.nan, dtype=np.float32)
    for ci, cond in enumerate(CONDITIONS):
        smask = np.ones(n, dtype=bool) if cond == "all" else (df["condition"].values == cond)
        for c in range(n_contact):
            rows = np.where(smask & (samp_cidx == c))[0]
            if rows.size:
                cube[ci, :, c, :] = np.nanmean(bandval[rows], axis=0)
        present = np.isfinite(cube[ci]).any(axis=(0, 2)).sum()
        print(f"[cube] cond={cond:8s} contacts_present={present}/{n_contact} samples={int(smask.sum())}")

    # ---- Projection (geometry only) --------------------------------------
    contact_xyz = merged[["x", "y", "z"]].to_numpy(dtype=np.float32)
    proj = {}
    for hemi in ("lh", "rh"):
        verts = hemi_vertices(hemi)
        idx, wraw = knn_gaussian(verts, contact_xyz, K_NEAREST, SIGMA_MM)
        proj[hemi] = (verts.shape[0], idx, wraw)
        print(f"[proj] {hemi}: nvert={verts.shape[0]} K={idx.shape[1]} meanW={wraw.mean():.4f}")

    # ---- Write bundle ----------------------------------------------------
    # cube as float16 to stay under GitHub's 100 MB/file limit (~53 MB vs ~106 MB);
    # plenty of precision for ERSP visualization. Decoded to float32 in the browser.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cube.astype("<f2").tofile(OUT_DIR / "cube_f16.bin")
    for hemi in ("lh", "rh"):
        _, idx, wraw = proj[hemi]
        idx.astype("<u2").tofile(OUT_DIR / f"proj_{hemi}_idx_u16.bin")
        wraw.astype("<f4").tofile(OUT_DIR / f"proj_{hemi}_w_f32.bin")

    # TN time axis: split 300 bins into the warped stimulus + response phases.
    _stim, _post = PROPORTIONS[1], PROPORTIONS[2]
    boundary = int(round(NT * _stim / (_stim + _post)))
    phases = [
        {"name": PHASE_NAMES[0], "start": 0, "end": boundary},
        {"name": PHASE_NAMES[1], "start": boundary, "end": int(NT)},
    ]
    contacts_meta = [
        {"patient_id": str(r.patient_id), "electrode": str(r.electrode),
         "x": float(r.x), "y": float(r.y), "z": float(r.z),
         "hemi": "lh" if str(r.hemi).upper().startswith("L") else "rh"}
        for r in merged.itertuples(index=False)
    ]
    (OUT_DIR / "contacts.json").write_text(json.dumps(contacts_meta), encoding="utf-8")

    manifest = {
        "source_run": str(ERSP_NPY.relative_to(REPO)).replace("\\", "/"),
        "coords_csv": str(COORDS_CSV.relative_to(REPO)).replace("\\", "/"),
        "conditions": CONDITIONS,
        "bands_hz": [list(b) for b in F_BANDS_HZ],
        "bands_bins": [list(b) for b in band_bins],
        "n_cond": n_cond, "n_band": n_band, "n_contact": n_contact, "n_time": int(NT),
        "mode": "TN", "proportions": list(PROPORTIONS),
        "phases": phases, "phase_boundary": boundary,
        "fmax_hz": FMAX_HZ, "k_nearest": int(K_NEAREST), "sigma_mm": float(SIGMA_MM),
        "agg_contact": AGG_CONTACT, "vmin": VMIN, "vmax": VMAX, "cmap": CMAP,
        "density_norm_q": DENSITY_NORM_Q,
        "alpha_min": ALPHA_MIN, "alpha_max": ALPHA_MAX, "alpha_gamma": ALPHA_GAMMA,
        "base_brain_gray_rgb": BASE_BRAIN_GRAY_RGB,
        "hemis": {
            # use .gii — the .mz3 from build_fsaverage_meshes.py carry a wrong magic
            # word and Niivue rejects them; .gii share the same vertex order.
            "lh": {"nvert": int(proj["lh"][0]), "mesh": "../meshes/fsaverage_lh.gii",
                   "mesh_inflated": "../meshes/fsaverage_lh.inflated.gii"},
            "rh": {"nvert": int(proj["rh"][0]), "mesh": "../meshes/fsaverage_rh.gii",
                   "mesh_inflated": "../meshes/fsaverage_rh.inflated.gii"},
        },
        "files": {
            "cube": {"name": "cube_f16.bin", "dtype": "float16",
                     "shape": [n_cond, n_band, n_contact, int(NT)]},
            "proj_lh_idx": {"name": "proj_lh_idx_u16.bin", "dtype": "uint16",
                            "shape": [int(proj["lh"][0]), int(K_NEAREST)]},
            "proj_lh_w": {"name": "proj_lh_w_f32.bin", "dtype": "float32",
                          "shape": [int(proj["lh"][0]), int(K_NEAREST)]},
            "proj_rh_idx": {"name": "proj_rh_idx_u16.bin", "dtype": "uint16",
                            "shape": [int(proj["rh"][0]), int(K_NEAREST)]},
            "proj_rh_w": {"name": "proj_rh_w_f32.bin", "dtype": "float32",
                          "shape": [int(proj["rh"][0]), int(K_NEAREST)]},
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_mb = sum(p.stat().st_size for p in OUT_DIR.glob("*")) / (1024 * 1024)
    print(f"\n[done] wrote bundle -> {OUT_DIR}  ({total_mb:.1f} MB)")
    for p in sorted(OUT_DIR.glob("*")):
        print(f"   {p.name:28s} {p.stat().st_size/1024:>10.0f} KB")


if __name__ == "__main__":
    main()
