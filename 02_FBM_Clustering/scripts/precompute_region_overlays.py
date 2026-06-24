"""
precompute_region_overlays.py — anatomical landmark overlays for the Activity Visualizer.

Curated language/auditory/visual/motor landmarks from the fsaverage Destrieux
(aparc.a2009s) parcellation, organized into FUNCTIONAL SYSTEMS. Each system has a
base hue; its sub-regions get distinct shades of that hue (e.g. Auditory = pinks).
Both hemispheres.

Outputs into the activity_viz bundle:
  - region_fill_{lh,rh}_u8.bin   per-vertex region id (0 = none, 1..N)
  - region_edge_{lh,rh}_u8.bin   region id on a thick boundary band (for outlines)
  - regions.json + manifest patch (id, name, system, color)

Run with the env that has numpy + nibabel (Python 3.11):
    "C:/Users/fanda/AppData/Local/Programs/Python/Python311/python.exe" \
        02_FBM_Clustering/scripts/precompute_region_overlays.py
"""

import colorsys
import gzip
import json
import struct
from pathlib import Path

import numpy as np
import nibabel as nib

REPO = Path(r"S:\HumanNeuronLab\ANALYSIS\FLM\Analysis_LoraFanda")
FS_LABEL = Path(r"S:\HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_HUG\fsaverage\label")
MESH_DIR = REPO / "02_FBM_Clustering" / "outputs" / "250_recon" / "fsaverage" / "meshes"
OUT_DIR = REPO / "02_FBM_Clustering" / "outputs" / "250_recon" / "fsaverage" / "activity_viz"
ANNOT = "aparc.a2009s"
EDGE_DILATE = 3   # boundary-band width in rings (thicker outline)

# Functional systems: (name, base_hue 0-1, [ (sub-region name, [Destrieux parcels]) ... ]).
# Sub-regions get shades of the system hue (light -> dark).
SYSTEMS = [
    ("Auditory (HG/STG)", 0.95, [
        ("Heschl's gyrus",      ["G_temp_sup-G_T_transv"]),
        ("Planum temporale",    ["G_temp_sup-Plan_tempo"]),
        ("Lateral STG",         ["G_temp_sup-Lateral"]),
    ]),
    ("Visual / VWFA", 0.78, [
        ("V1 (calcarine)",      ["S_calcarine"]),
        ("VWFA / fusiform",     ["G_oc-temp_lat-fusifor"]),
        ("Lingual",             ["G_oc-temp_med-Lingual"]),
        ("Lateral occipital",   ["G_occipital_middle"]),
    ]),
    ("Temporal lexical-semantic", 0.46, [
        ("MTG",                 ["G_temporal_middle"]),
        ("ITG",                 ["G_temporal_inf"]),
        ("ATL / temporal pole", ["Pole_temporal"]),
    ]),
    ("Frontal (IFG / Broca)", 0.60, [
        ("Pars opercularis",    ["G_front_inf-Opercular"]),
        ("Pars triangularis",   ["G_front_inf-Triangul"]),
        ("Pars orbitalis",      ["G_front_inf-Orbital"]),
    ]),
    ("Speech-motor (sensorimotor)", 0.08, [
        ("M1 (precentral)",     ["G_precentral"]),
        ("S1 (postcentral)",    ["G_postcentral"]),
        ("Central sulcus",      ["S_central"]),
    ]),
]


def shade(hue, i, n):
    """RGB 0-255 for sub-region i of n within a hue family (vary lightness + a little sat)."""
    l = 0.68 - 0.30 * (i / max(1, n - 1))          # light -> dark
    s = 0.85 - 0.10 * (i / max(1, n - 1))
    r, g, b = colorsys.hls_to_rgb(hue, l, s)
    return [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]


def read_mz3(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic, attr, nface, nvert, nskip = struct.unpack_from("<HHIII", raw, 0)
    off = 16 + nskip
    faces = np.frombuffer(raw, dtype="<u4", count=nface * 3, offset=off).reshape(nface, 3)
    return int(nvert), np.ascontiguousarray(faces)


def build_hemi(hemi, parcel_to_id):
    labels, _, names = nib.freesurfer.read_annot(str(FS_LABEL / f"{hemi}.{ANNOT}.annot"))
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    nvert, faces = read_mz3(MESH_DIR / f"fsaverage_{hemi}.mz3")
    assert labels.shape[0] == nvert

    fill = np.zeros(nvert, dtype=np.uint8)
    for parcel, rid in parcel_to_id.items():
        if parcel in names:
            fill[labels == names.index(parcel)] = rid

    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    a, b = e[:, 0], e[:, 1]
    diff = fill[a] != fill[b]
    edge = np.zeros(nvert, dtype=np.uint8)
    for u, w in ((a, b), (b, a)):
        m = diff & (fill[u] != 0)
        edge[u[m]] = fill[u[m]]
    for _ in range(EDGE_DILATE):
        grow = np.zeros(nvert, dtype=np.uint8)
        for u, w in ((a, b), (b, a)):
            m = (edge[w] != 0) & (fill[u] == fill[w]) & (edge[u] == 0)
            grow[u[m]] = fill[u[m]]
        edge = np.maximum(edge, grow)
    return fill, edge


def main():
    # flatten systems -> regions with ids + shaded colors
    regions, parcel_to_id = [], {}
    rid = 0
    for sysname, hue, subs in SYSTEMS:
        for i, (rname, parcels) in enumerate(subs):
            rid += 1
            col = shade(hue, i, len(subs))
            regions.append({"id": rid, "name": rname, "system": sysname, "color": col})
            for p in parcels:
                parcel_to_id[p] = rid

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for hemi in ("lh", "rh"):
        fill, edge = build_hemi(hemi, parcel_to_id)
        fill.tofile(OUT_DIR / f"region_fill_{hemi}_u8.bin")
        edge.tofile(OUT_DIR / f"region_edge_{hemi}_u8.bin")
        print(f"[{hemi}] fill={int((fill>0).sum())}  edge={int((edge>0).sum())}")

    (OUT_DIR / "regions.json").write_text(json.dumps({"regions": regions}, indent=2), encoding="utf-8")
    man_path = OUT_DIR / "manifest.json"
    man = json.loads(man_path.read_text())
    man["regions"] = regions
    man["region_files"] = {
        "fill_lh": "region_fill_lh_u8.bin", "fill_rh": "region_fill_rh_u8.bin",
        "edge_lh": "region_edge_lh_u8.bin", "edge_rh": "region_edge_rh_u8.bin",
    }
    man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"[done] {len(regions)} regions in {len(SYSTEMS)} systems -> {OUT_DIR}")
    for r in regions:
        print(f"   {r['id']:2d} [{r['system']:28s}] {r['name']:22s} rgb{tuple(r['color'])}")


if __name__ == "__main__":
    main()
