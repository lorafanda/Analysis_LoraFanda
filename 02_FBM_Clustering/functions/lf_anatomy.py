"""
lf_anatomy.py — Per-electrode anatomical labels + per-cluster purity.

For each electrode in `outputs/250_recon/fsaverage/coords/*_contacts_fsaverage.csv`,
projects (x, y, z) MNI coordinates to the nearest fsaverage cortical surface
vertex and reads the Desikan-Killiany aparc label of that vertex.

Used by the validation pipeline (211_validation.ipynb) to compute per-cluster
anatomical purity:

  - top_3 regions per cluster + their proportions
  - entropy of the aparc-label distribution (low = anatomically coherent)
  - purity = proportion of the modal region
  - permutation null: shuffle labels N times, recompute mean entropy ->
    p-value for "is this cluster more anatomically coherent than chance?"

References:
  Hamilton, L.S., Edwards, E. & Chang, E.F. (2018).
    A spatial map of onset and sustained responses to speech in the human
    superior temporal gyrus. Current Biology 28(12), 1860–1871.
  Forseth, K.J. et al. (2018).
    A lexical semantic hub for heteromodal naming in middle fusiform gyrus.
    Brain 141(7), 2112–2126.

Public API:
  build_aparc_cache(coords_csv_glob, output_csv, subjects_dir=None)
      -> walks all *_contacts_fsaverage.csv files, projects each electrode
         to fsaverage cortex, looks up aparc label, writes a single CSV
         cache mapping (patient, electrode) -> aparc_label.

  cluster_anatomy_purity(df_labels, df_aparc, *, n_perm=1000)
      -> per-cluster dict with top_3, entropy, purity, p_value.

Requires: mne (with fsaverage subjects_dir available — fetched lazily
on first call via mne.datasets.fetch_fsaverage()).
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Aparc cache (one-time precompute)
# ============================================================
def build_aparc_cache(
    coords_csv_glob,
    output_csv,
    *,
    subjects_dir: Optional[Path] = None,
    annot_parc: str = "aparc",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Walk every *_contacts_fsaverage.csv under `coords_csv_glob`, read each
    electrode's (x, y, z), project to the nearest fsaverage cortical vertex
    (combined lh+rh white surface), look up the aparc label, write a single
    cache CSV.

    Schema of the cache:
        patient, electrode, hemi, aparc_label, vertex_idx, x, y, z

    `subjects_dir` defaults to MNE's installed fsaverage; if missing,
    fetched via mne.datasets.fetch_fsaverage(). On first run this may
    download ~50MB; subsequent runs reuse the local copy.

    Cache build is O(n_electrodes × n_surface_vertices) per nearest-vertex
    search; for ~5000 electrodes × ~150k vertices this takes ~30s on a
    laptop. Cache is written once, then loaded as a flat CSV from then on.
    """
    import mne

    # Resolve subjects_dir
    if subjects_dir is None:
        try:
            subjects_dir = Path(mne.datasets.fetch_fsaverage(verbose=False)).parent
        except Exception as e:
            raise RuntimeError(
                "Couldn't locate or fetch fsaverage. "
                "Set SUBJECTS_DIR env var, or pass subjects_dir explicitly."
            ) from e
    subjects_dir = Path(subjects_dir)

    if verbose:
        print(f"[aparc_cache] subjects_dir = {subjects_dir}")
        print(f"[aparc_cache] parc = {annot_parc}")

    # Load fsaverage pial coords + aparc labels for both hemis
    lh_pial = mne.read_surface(subjects_dir / "fsaverage" / "surf" / "lh.pial")[0]
    rh_pial = mne.read_surface(subjects_dir / "fsaverage" / "surf" / "rh.pial")[0]

    lh_labels = mne.read_labels_from_annot("fsaverage", parc=annot_parc, hemi="lh",
                                            subjects_dir=str(subjects_dir), verbose=False)
    rh_labels = mne.read_labels_from_annot("fsaverage", parc=annot_parc, hemi="rh",
                                            subjects_dir=str(subjects_dir), verbose=False)

    # Build vertex -> label arrays per hemi
    n_lh = lh_pial.shape[0]
    n_rh = rh_pial.shape[0]
    lh_vert_label = np.full(n_lh, "unknown", dtype=object)
    rh_vert_label = np.full(n_rh, "unknown", dtype=object)
    for lbl in lh_labels:
        for v in lbl.vertices:
            lh_vert_label[int(v)] = lbl.name.replace("-lh", "")
    for lbl in rh_labels:
        for v in lbl.vertices:
            rh_vert_label[int(v)] = lbl.name.replace("-rh", "")

    # Walk all coords CSVs
    coords_dir = Path(coords_csv_glob).parent
    pattern = Path(coords_csv_glob).name
    files = sorted(coords_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No coords CSV matched {coords_csv_glob}")

    if verbose:
        print(f"[aparc_cache] found {len(files)} coords CSV files")

    rows = []
    for f in files:
        df = pd.read_csv(f)
        # Skip the ALL_PATIENTS aggregate file (rebuilt from per-patient files)
        if "ALL_PATIENTS" in f.name.upper():
            continue
        if not {"patient", "name", "x", "y", "z"}.issubset(df.columns):
            if verbose:
                print(f"  [skip] {f.name}: missing required columns")
            continue
        for _, r in df.iterrows():
            x, y, z = float(r["x"]), float(r["y"]), float(r["z"])
            hemi = str(r.get("hemi", "L" if x < 0 else "R")).upper()
            if hemi.startswith("L"):
                d2 = ((lh_pial - [x, y, z]) ** 2).sum(axis=1)
                v = int(np.argmin(d2))
                label = str(lh_vert_label[v])
                hemi_norm = "lh"
            else:
                d2 = ((rh_pial - [x, y, z]) ** 2).sum(axis=1)
                v = int(np.argmin(d2))
                label = str(rh_vert_label[v])
                hemi_norm = "rh"
            rows.append({
                "patient": r["patient"],
                "electrode": r["name"],
                "hemi": hemi_norm,
                "aparc_label": label,
                "vertex_idx": v,
                "x": x, "y": y, "z": z,
            })

    df_out = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    if verbose:
        print(f"[aparc_cache] wrote {len(df_out)} rows -> {output_csv}")
        print(df_out["aparc_label"].value_counts().head(15))
    return df_out


# ============================================================
# Per-cluster anatomy purity + permutation test
# ============================================================
def _entropy(counts):
    """Shannon entropy in bits."""
    total = sum(counts)
    if total == 0:
        return float("nan")
    H = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        H -= p * math.log2(p)
    return H


def cluster_anatomy_purity(
    df_labels: pd.DataFrame,
    df_aparc: pd.DataFrame,
    *,
    cluster_col: str = "cluster",
    patient_col: str = "patient_id",
    electrode_col: str = "electrode",
    n_perm: int = 0,
    random_state: int = 0,
) -> Dict[int, Dict]:
    """
    For each cluster in df_labels, compute its aparc-label distribution
    (joined from df_aparc on (patient, electrode)) and summarize:

        top_3     : list of (aparc_label, proportion) — 3 most common regions
        entropy   : Shannon entropy in bits (low = anatomically coherent)
        purity    : proportion of the modal region (0..1, high = coherent)
        n_total   : samples in cluster
        n_unknown : samples whose electrode wasn't found in df_aparc
        p_value   : permutation p (omitted when n_perm == 0); fraction of
                    null-entropies <= observed (lower p = more coherent than chance)

    df_aparc must have columns: patient, electrode, aparc_label
    df_labels must have the cluster col + patient_col + electrode_col.
    """
    # Build a fast lookup
    df_aparc = df_aparc.copy()
    df_aparc["patient"] = df_aparc["patient"].astype(str)
    df_aparc["electrode"] = df_aparc["electrode"].astype(str)
    lookup = {
        (str(p), str(e)): str(lbl)
        for p, e, lbl in zip(df_aparc["patient"], df_aparc["electrode"], df_aparc["aparc_label"])
    }

    df_l = df_labels.copy()
    df_l["aparc_label"] = [
        lookup.get((str(p), str(e)), "unknown")
        for p, e in zip(df_l[patient_col], df_l[electrode_col])
    ]

    out: Dict[int, Dict] = {}
    rng = np.random.default_rng(random_state)
    all_aparcs = df_l["aparc_label"].to_numpy()

    for c in sorted(df_l[cluster_col].unique()):
        sub = df_l[df_l[cluster_col] == c]["aparc_label"]
        counts = Counter(sub)
        n_total = int(len(sub))
        n_unknown = int(counts.get("unknown", 0))
        # Drop 'unknown' from purity/entropy unless it's the only thing
        known_counts = {k: v for k, v in counts.items() if k != "unknown"}
        if not known_counts:
            entropy = float("nan")
            purity = float("nan")
            top_3 = [("unknown", 1.0)]
        else:
            total_known = sum(known_counts.values())
            entropy = _entropy(list(known_counts.values()))
            purity  = max(known_counts.values()) / total_known
            sorted_counts = sorted(known_counts.items(), key=lambda kv: -kv[1])[:3]
            top_3 = [(lbl, round(cnt / total_known, 3)) for lbl, cnt in sorted_counts]

        entry = {
            "cluster_id": int(c),
            "n_total": n_total,
            "n_unknown": n_unknown,
            "top_3": top_3,
            "entropy_bits": float(entropy),
            "purity": float(purity),
        }

        if n_perm > 0 and not math.isnan(entropy):
            # Sample n_total aparc labels uniformly from the WHOLE pool
            # (without replacement), recompute entropy. Repeat n_perm times.
            null_entropies = np.empty(n_perm, dtype=np.float64)
            pool = all_aparcs[all_aparcs != "unknown"]
            for b in range(n_perm):
                if len(pool) < n_total:
                    null_entropies[b] = float("nan"); continue
                samp = rng.choice(pool, size=n_total, replace=False)
                cnt = Counter(samp)
                null_entropies[b] = _entropy(list(cnt.values()))
            # p = P(null_entropy <= observed)
            valid = ~np.isnan(null_entropies)
            entry["p_entropy_lower_tail"] = float((null_entropies[valid] <= entropy).mean()) if valid.any() else float("nan")
            entry["null_entropy_mean"] = float(np.nanmean(null_entropies))

        out[int(c)] = entry

    return out


def save_anatomy_artifacts(
    run_dir,
    df_labels: pd.DataFrame,
    df_aparc: pd.DataFrame,
    *,
    cluster_col: str = "cluster",
    n_perm: int = 0,
    verbose: bool = True,
):
    """Compute + save per-cluster anatomy purity into run_dir."""
    run_dir = Path(run_dir)
    res = cluster_anatomy_purity(df_labels, df_aparc, cluster_col=cluster_col, n_perm=n_perm)

    rows = []
    for c, entry in res.items():
        rows.append({
            "cluster_id": entry["cluster_id"],
            "n_total":    entry["n_total"],
            "n_unknown":  entry["n_unknown"],
            "entropy_bits": entry["entropy_bits"],
            "purity":     entry["purity"],
            "top_region": entry["top_3"][0][0] if entry["top_3"] else "",
            "top_proportion": entry["top_3"][0][1] if entry["top_3"] else float("nan"),
            "top_3_json": json.dumps(entry["top_3"]),
            **({"p_entropy_lower_tail": entry.get("p_entropy_lower_tail"),
                "null_entropy_mean":   entry.get("null_entropy_mean")} if n_perm else {}),
        })
    df_out = pd.DataFrame(rows).sort_values("cluster_id")
    df_out.to_csv(run_dir / "per_cluster_anatomy.csv", index=False)
    (run_dir / "per_cluster_anatomy.json").write_text(json.dumps(res, indent=2))

    if verbose:
        print(f"[anatomy] {len(df_out)} clusters scored, wrote {run_dir / 'per_cluster_anatomy.csv'}")
        print(df_out[["cluster_id", "n_total", "purity", "entropy_bits", "top_region", "top_proportion"]].to_string(index=False))

    return df_out, res


# ============================================================
# Spatial compactness (mirrored-XYZ pairwise distance)
# ============================================================
# Used by 213_cluster_ranking.ipynb as the "anatomy" axis instead of purity.
# Rationale: in iEEG/SEEG, functionally related responses typically come from
# physically nearby brain regions. The aparc-label purity score is one proxy
# (same Desikan parcel) but it's coarse: two contacts in the same parcel might
# be 30 mm apart, and two contacts in adjacent parcels might be 5 mm apart.
# A direct Euclidean distance in fsaverage space is sharper.
#
# Bilateral homology: language / sensorimotor / memory networks span both
# hemispheres in roughly mirror-symmetric layouts. So an iEEG cluster that
# bridges left/right STG with no contralateral surprise should be just as
# "compact" as a unilateral one. We mirror every contact to one hemisphere
# (default: left, via x_new = -|x|) before computing distance, which folds
# bilateral homotopic contacts onto each other.

def cluster_spatial_compactness(
    df_labels: pd.DataFrame,
    df_coords: pd.DataFrame,
    *,
    cluster_col:   str = "cluster",
    patient_col:   str = "patient_id",
    electrode_col: str = "electrode",
    mirror_to:     str = "left",
    metric:        str = "mean_pairwise",
    verbose:       bool = False,
) -> Dict[int, Dict]:
    """
    Per-cluster spatial compactness in fsaverage / MNI mm.

    Parameters
    ----------
    df_labels : DataFrame with patient_col + electrode_col + cluster_col
        (e.g. labels.csv from a clustering run).
    df_coords : DataFrame with 'patient' + ('electrode' or 'name') + 'x','y','z'
        columns. The aparc cache (`aparc_lookup.csv`) is a fine input — it
        already carries the per-contact fsaverage coords.
    mirror_to : 'left' (x_new = -|x|, default), 'right' (x_new = |x|),
        or 'none' (no mirror — for sanity checks).
    metric :
        'mean_pairwise'   — mean Euclidean distance over all (i, j) pairs.
                            Canonical compactness measure. Default.
        'median_pairwise' — robust to one or two outliers.
        'max_pairwise'    — diameter of the cluster.
        'mean_centroid'   — radius-of-gyration: mean distance from centroid.

    Returns
    -------
    Dict[int cluster_id, {
        "cluster_id":    int
        "n_total":       int    samples in the cluster
        "n_with_coords": int    samples successfully joined to df_coords
        "n_missing":     int    samples whose electrode wasn't in df_coords
        "metric":        str
        "mirror_to":     str
        "distance_mm":   float  the compactness value (LOWER = tighter)
        "centroid_xyz":  [x, y, z]  centroid AFTER mirroring
        "spread_xyz":    [std_x, std_y, std_z]  per-axis std after mirroring
    }]

    NaN distance is used when a cluster has fewer than 2 coord-resolved
    members (the metric is undefined there).
    """
    from scipy.spatial.distance import pdist

    df_l = df_labels.copy()
    df_l[patient_col]   = df_l[patient_col].astype(str)
    df_l[electrode_col] = df_l[electrode_col].astype(str)

    # Build (patient, electrode-or-name) -> (x, y, z) lookup
    df_c = df_coords.copy()
    pat_c  = "patient" if "patient" in df_c.columns else patient_col
    name_c = "electrode" if "electrode" in df_c.columns else ("name" if "name" in df_c.columns else None)
    if name_c is None:
        raise ValueError("df_coords must have 'electrode' or 'name' column")
    df_c[pat_c]  = df_c[pat_c].astype(str)
    df_c[name_c] = df_c[name_c].astype(str)
    coord_lookup = {}
    for p, n, x, y, z in zip(df_c[pat_c], df_c[name_c], df_c["x"], df_c["y"], df_c["z"]):
        try:
            coord_lookup[(str(p), str(n))] = (float(x), float(y), float(z))
        except (TypeError, ValueError):
            continue

    def _mirror(xyz):
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        if mirror_to == "left":   return (-abs(x), y, z)
        if mirror_to == "right":  return ( abs(x), y, z)
        return (x, y, z)

    out: Dict[int, Dict] = {}
    for c in sorted(df_l[cluster_col].unique()):
        sub = df_l[df_l[cluster_col] == c]
        n_total = int(len(sub))
        coords = []
        for pid, elec in zip(sub[patient_col], sub[electrode_col]):
            xyz = coord_lookup.get((str(pid), str(elec)))
            if xyz is not None:
                coords.append(_mirror(xyz))
        n_with    = len(coords)
        n_missing = n_total - n_with

        entry = {
            "cluster_id":    int(c),
            "n_total":       n_total,
            "n_with_coords": n_with,
            "n_missing":     n_missing,
            "metric":        metric,
            "mirror_to":     mirror_to,
            "distance_mm":   float("nan"),
            "centroid_xyz":  [float("nan")] * 3,
            "spread_xyz":    [float("nan")] * 3,
        }

        if n_with >= 2:
            P = np.asarray(coords, dtype=np.float64)        # (n, 3)
            entry["centroid_xyz"] = P.mean(axis=0).tolist()
            entry["spread_xyz"]   = P.std(axis=0).tolist()

            if metric in ("mean_pairwise", "median_pairwise", "max_pairwise"):
                d = pdist(P, metric="euclidean")
                if metric == "mean_pairwise":
                    entry["distance_mm"] = float(d.mean())
                elif metric == "median_pairwise":
                    entry["distance_mm"] = float(np.median(d))
                elif metric == "max_pairwise":
                    entry["distance_mm"] = float(d.max())
            elif metric == "mean_centroid":
                centroid = np.array(entry["centroid_xyz"])
                d_centroid = np.linalg.norm(P - centroid, axis=1)
                entry["distance_mm"] = float(d_centroid.mean())
            else:
                raise ValueError(f"unknown metric: {metric}")
        elif n_with == 1:
            entry["centroid_xyz"] = list(coords[0])

        if verbose:
            print(f"  cluster {c}: n={n_total} with_coords={n_with} "
                  f"distance={entry['distance_mm']:.2f}mm")
        out[int(c)] = entry

    return out


def save_spatial_compactness_artifacts(
    run_dir,
    df_labels:   pd.DataFrame,
    df_coords:   pd.DataFrame,
    *,
    cluster_col: str = "cluster",
    mirror_to:   str = "left",
    metric:      str = "mean_pairwise",
    verbose:     bool = True,
):
    """
    Compute + save per-cluster spatial compactness for a clustering run.

    Writes:
        per_cluster_spatial_compactness.csv    (cluster_id, n_total,
            n_with_coords, n_missing, distance_mm, centroid_{x,y,z},
            spread_{x,y,z}, metric, mirror_to)
        per_cluster_spatial_compactness.json   (full dict, mirrors the
            CSV but keeps numpy-list fields as lists for downstream JS)

    Returns (df, full_dict).
    """
    run_dir = Path(run_dir)
    res = cluster_spatial_compactness(
        df_labels, df_coords,
        cluster_col=cluster_col,
        mirror_to=mirror_to,
        metric=metric,
        verbose=verbose,
    )

    rows = []
    for c, entry in res.items():
        cx, cy, cz = entry["centroid_xyz"]
        sx, sy, sz = entry["spread_xyz"]
        rows.append({
            "cluster_id":    entry["cluster_id"],
            "n_total":       entry["n_total"],
            "n_with_coords": entry["n_with_coords"],
            "n_missing":     entry["n_missing"],
            "distance_mm":   entry["distance_mm"],
            "centroid_x":    cx, "centroid_y": cy, "centroid_z": cz,
            "spread_x":      sx, "spread_y":   sy, "spread_z":   sz,
            "metric":        entry["metric"],
            "mirror_to":     entry["mirror_to"],
        })
    df_out = pd.DataFrame(rows).sort_values("cluster_id")
    df_out.to_csv(run_dir / "per_cluster_spatial_compactness.csv", index=False)
    (run_dir / "per_cluster_spatial_compactness.json").write_text(
        json.dumps(res, indent=2, default=str)
    )

    if verbose:
        finite = df_out["distance_mm"].dropna()
        if len(finite):
            print(f"[spatial] {len(df_out)} clusters · metric={metric} · mirror={mirror_to} · "
                  f"median {finite.median():.1f}mm  min {finite.min():.1f}  max {finite.max():.1f}")
        else:
            print(f"[spatial] {len(df_out)} clusters · no finite distances")
        print(df_out[["cluster_id", "n_total", "n_with_coords", "distance_mm"]].to_string(index=False))

    return df_out, res
