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
    from mne.surface import _read_surface_meas as _maybe_unused  # noqa: F401

    # Resolve subjects_dir
    if subjects_dir is None:
        try:
            subjects_dir = mne.datasets.fetch_fsaverage(verbose=False).parent
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
