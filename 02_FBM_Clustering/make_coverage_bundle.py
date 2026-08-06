#!/usr/bin/env python3
"""
make_coverage_bundle.py — per-vertex sampling counts for clustering_visualizer.html.

The question the page answers: at this point on the cortex, HOW WELL DID WE SAMPLE, and
of the electrodes that are here, what fraction belong to a given cluster? Every brain
figure in this project carries the caveat "dot density reflects where electrodes were
implanted, not where language is" — this makes that checkable instead of a footnote.

Why not reuse the activity-visualizer projection: that one keeps the k=4 NEAREST
contacts per vertex, so a patient count taken from it can never exceed 4 of 35 and is
useless as a probability. Here every contact within RADIUS_MM of a vertex is counted,
which is what makes "fraction of patients sampled here" meaningful.

LAYOUT — the counts split into two kinds, because they have different lifetimes:

    coverage_{lh,rh}_u8.bin      n_contacts, n_patients      SHARED by every run
    runs/<id>/clusters_{h}_u8.bin  n_cluster_0 .. n_cluster_{K-1}   per run

Sampling depends only on where the electrodes are, so it is identical across runs and
is stored once. Only the cluster breakdown is per-run. That is what makes shipping many
runs affordable (~1.6 MB per run instead of ~4.6 MB).

uint8 is checked, not assumed: if any count exceeds 255 the file is written as uint16
and the manifest says so, so a larger radius cannot silently wrap around.

Per vertex this gives the page everything it needs to compute both maps as plain ratios:

    coverage   P(sampled)          = n_patients / n_patients_total
    cluster k  P(cluster k | here) = n_cluster_k / (labelled contacts here)

Counts, not probabilities, are shipped on purpose: the ratio you want depends on the
question, and a stored ratio hides its own denominator.

    python make_coverage_bundle.py                 # every run in every known track
    python make_coverage_bundle.py --radius 8
    python make_coverage_bundle.py --run <run dir> # just one
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FS = ROOT / "outputs" / "250_recon" / "fsaverage"
MESHES = FS / "meshes"
COORDS = FS / "coords" / "ALL_PATIENTS_contacts_fsaverage.csv"
OUT = FS / "coverage_viz"
CLUSTERING = ROOT / "outputs" / "clustering"
RADIUS_MM = 10.0

# Tracks offered in the page's dropdown, in display order. Only the NEWEST run per track
# is bundled by default (--per-track raises it): within a track the older runs are
# superseded predecessors on smaller cohorts, and mixing cohorts silently is the exact
# problem the 1027-electrode cleanup fixed.
#
# Deliberately absent: every *_raw track (~6 GB, no consensus/ranking/stability, three of
# them pinned at K=20 = the sweep ceiling) and hierarchical/concat_rawds (K=20 for the
# same reason, and the largest payload of the set).
TRACKS = [
    ("kmeans/concat_hg", "K-means · concatenated", "1027-electrode concat cohort"),
    ("hierarchical/concat_hg", "Ward · concatenated", "1027-electrode concat cohort"),
    ("kmeans/concat_rawds", "K-means · concat 15-band", "1027-electrode concat cohort"),
    ("kmeans/hg", "K-means · per-condition", "2026-07-19 per-task cohort"),
    ("hierarchical/hg", "Ward · per-condition", "2026-07-19 per-task cohort"),
    ("kmeans/rawds", "K-means · per-condition 15-band", "2026-07-19 per-task cohort"),
    ("hierarchical/rawds", "Ward · per-condition 15-band", "2026-07-19 per-task cohort"),
]

# runs known to be unusable; see the scout notes in the commit message
SKIP_RUNS = {
    "20260731_174851",   # no recon/ at all
    "20260802_220854",   # empty dir
    "20260525_171725",   # only Thumbs.db
    "20260728_110801",   # 10 orphan recon dirs from a stale K=10 render, but K=5
}


def norm(s) -> str:
    return "" if s is None else str(s).replace("_", "").replace("-", "").upper()


def write_counts(arr: np.ndarray, path_stem: Path) -> tuple[str, str]:
    """Write as uint8 when it fits, uint16 otherwise. Returns (filename, dtype)."""
    if int(arr.max()) < 256:
        f = path_stem.with_name(path_stem.name + "_u8.bin")
        arr.astype(np.uint8).tofile(f)
        return f.name, "uint8"
    f = path_stem.with_name(path_stem.name + "_u16.bin")
    arr.astype(np.uint16).tofile(f)
    return f.name, "uint16"


def discover_runs(explicit: str | None, per_track: int) -> list[tuple[str, str, str, Path]]:
    """-> [(track_key, track_label, cohort, run_dir), ...] newest first within each track."""
    meta = {k: (lbl, coh) for k, lbl, coh in TRACKS}
    if explicit:
        d = Path(explicit).resolve()
        try:
            key = "/".join(d.relative_to(CLUSTERING).parts[:2])
        except ValueError:
            key = "custom"
        lbl, coh = meta.get(key, (key, ""))
        return [(key, lbl, coh, d)]

    found = []
    for key, label, cohort in TRACKS:
        rd = CLUSTERING / key / "runs"
        if not rd.is_dir():
            continue
        kept = 0
        for d in sorted((p for p in rd.iterdir() if p.is_dir()),
                        key=lambda p: p.name, reverse=True):
            if d.name in SKIP_RUNS or not (d / "labels.csv").exists():
                continue
            found.append((key, label, cohort, d))
            kept += 1
            if kept >= per_track:
                break
    return found


def git_tracked() -> set[str]:
    """Paths git actually has, as repo-relative posix strings.

    The page fetches figures over raw.githubusercontent, so a file that exists on this
    share but was never committed is a 404 waiting to happen. Advertising only tracked
    files turns that into a figure that is absent from the PDF rather than a broken one.
    An empty set (git unavailable) disables the filter rather than dropping everything.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["git", "ls-files", "--", "02_FBM_Clustering/outputs/clustering"],
            cwd=ROOT.parent, capture_output=True, text=True, timeout=120, check=True).stdout
        return set(out.splitlines())
    except Exception as e:                                   # noqa: BLE001
        print(f"  (git ls-files unavailable: {e}; shipping all figures found on disk)")
        return set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=RADIUS_MM)
    ap.add_argument("--run", default=None, help="one clustering run dir (default: newest per track)")
    ap.add_argument("--per-track", type=int, default=1, help="how many runs per track to bundle")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import nibabel as nib
    from scipy.spatial import cKDTree

    runs = discover_runs(a.run, a.per_track)
    tracked = git_tracked()
    if not runs:
        print("  no runs found", file=sys.stderr)
        return 1
    print(f"  {len(runs)} run(s) across {len({k for k, _, _, _ in runs})} track(s)")

    co = pd.read_csv(COORDS).dropna(subset=["x", "y", "z"])
    co["key"] = [f"{p}|{norm(n)}" for p, n in zip(co["patient"], co["name"])]
    n_pat_total = int(co["patient"].nunique())
    pat_codes = {p: i for i, p in enumerate(sorted(co["patient"].unique()))}

    # ── shared: sampling geometry. Identical for every run, so computed once.
    hemis, near_by_hemi, sub_by_hemi = {}, {}, {}
    for h, tag in (("lh", "L"), ("rh", "R")):
        verts = nib.load(str(MESHES / f"fsaverage_{h}.gii")).darrays[0].data.astype(float)
        sub = co[co["hemi"].astype(str).str.upper().str[0] == tag].reset_index(drop=True)
        near = cKDTree(sub[["x", "y", "z"]].to_numpy()).query_ball_point(verts, r=a.radius)
        near_by_hemi[h], sub_by_hemi[h] = near, sub

        cov = np.zeros((len(verts), 2), dtype=np.int32)
        pcodes = sub["patient"].map(pat_codes).to_numpy()
        for v, hits in enumerate(near):
            if hits:
                cov[v, 0] = len(hits)
                cov[v, 1] = len(set(pcodes[hits]))
        fn, dt = write_counts(cov, OUT / f"coverage_{h}")
        pct = 100 * float((cov[:, 1] > 0).mean())
        hemis[h] = {"nvert": int(len(verts)), "file": fn, "dtype": dt,
                    "n_contacts_hemi": int(len(sub)),
                    "pct_vertices_sampled": round(pct, 1),
                    "max_patients_at_a_vertex": int(cov[:, 1].max()),
                    "max_contacts_at_a_vertex": int(cov[:, 0].max())}
        print(f"  shared {h}: {len(verts)} verts, {len(sub)} contacts | {pct:.1f}% sampled | "
              f"{dt} | {(OUT / fn).stat().st_size/1e6:.2f} MB")

    # ── per run: only the cluster breakdown
    entries, total_mb = [], 0.0
    for key, label, cohort, run in runs:
        try:
            lab = pd.read_csv(run / "labels.csv")
            ccol = next(c for c in lab.columns
                        if c.startswith("cluster_") and not c.endswith("_ranked"))
        except Exception as e:            # noqa: B014 - any unreadable run is skipped
            print(f"  !! skip {key}/{run.name}: {e}")
            continue

        lab["key"] = [f"{p}|{norm(e)}" for p, e in zip(lab["patient_id"], lab["electrode"])]
        clusters = sorted(int(c) for c in lab[ccol].dropna().unique())
        if not clusters:
            print(f"  !! skip {key}/{run.name}: no cluster labels")
            continue
        cidx = {k: i for i, k in enumerate(clusters)}

        # ONE ELECTRODE CAN CARRY SEVERAL LABELS. In the per-condition tracks a row is an
        # electrode x condition sample, and 28% of electrodes land in a different cluster
        # for audio than for reading. Collapsing to one label per electrode (dict(zip(...))
        # keeps whichever row came last) would silently discard that, so every row counts
        # and the denominator becomes samples-near-here, which the manifest states.
        clu_of = lab.dropna(subset=[ccol]).groupby("key")[ccol].apply(
            lambda s: [int(v) for v in s]).to_dict()
        cl_all = co["key"].map(clu_of)
        n_rows_per_el = lab.groupby("key")[ccol].size()
        multi = int((lab.groupby("key")[ccol].nunique() > 1).sum())
        unit = "electrode" if int(n_rows_per_el.max()) == 1 else "electrode x condition"

        rdir = OUT / "runs" / f"{key.replace('/', '__')}__{run.name}"
        rdir.mkdir(parents=True, exist_ok=True)
        files, mb = {}, 0.0
        for h, tag in (("lh", "L"), ("rh", "R")):
            sub, near = sub_by_hemi[h], near_by_hemi[h]
            cl = cl_all[co["hemi"].astype(str).str.upper().str[0] == tag].to_numpy()
            arr = np.zeros((hemis[h]["nvert"], len(clusters)), dtype=np.int32)
            for v, hits in enumerate(near):
                for i in hits:
                    labels = cl[i]
                    if isinstance(labels, list):
                        for c in labels:
                            arr[v, cidx[c]] += 1
            fn, dt = write_counts(arr, rdir / f"clusters_{h}")
            files[h] = {"file": fn, "dtype": dt}
            mb += (rdir / fn).stat().st_size / 1e6

        matched = int(cl_all.notna().sum())
        n_samples = int(sum(len(v) for v in cl_all.dropna()))
        entries.append({
            "id": f"{key.replace('/', '__')}__{run.name}",
            "track": key, "track_label": label, "cohort": cohort, "run": run.name,
            "dir": f"runs/{key.replace('/', '__')}__{run.name}",
            "k": len(clusters), "clusters": clusters,
            "n_electrodes": int(len(lab)),
            "n_contacts_with_cluster": matched,
            "n_samples_with_cluster": n_samples,
            "unit": unit,
            "n_electrodes_multilabel": multi,
            "cluster_sizes": {str(k): int((lab[ccol] == k).sum()) for k in clusters},
            "label": f"{label} · {run.name} · K={len(clusters)} · n={len(lab)}",
            "hemis": files,
            "figures": run_figures(run, clusters, tracked, len(clusters)),
            "path": f"{key}/runs/{run.name}",
        })
        total_mb += mb
        extra = f", {multi} multi-label" if multi else ""
        print(f"  {key}/{run.name}: K={len(clusters)}, {len(lab)} rows [{unit}], "
              f"{matched} contacts / {n_samples} samples matched{extra} | {mb:.2f} MB")

    manifest = {
        "radius_mm": a.radius,
        "n_patients_total": n_pat_total,
        "n_contacts_total": int(len(co)),
        "coords": COORDS.name,
        "coverage_fields": ["n_contacts", "n_patients"],
        "hemis": hemis,
        "contacts_file": "contacts.json",
        "runs": entries,
        "note": ("counts within radius_mm of each vertex; ratios are computed client-side "
                 "so the denominator stays visible"),
    }

    # contact list for the electrode overlay, with each run's label so the page can
    # colour/filter contacts without refetching anything
    cj = {"xyz": [[round(float(r.x), 1), round(float(r.y), 1), round(float(r.z), 1)] for r in co.itertuples()],
          "hemi": ["lh" if str(r.hemi).upper().startswith("L") else "rh" for r in co.itertuples()],
          "runs": {}}
    for key, label, cohort, run in runs:
        try:
            lab = pd.read_csv(run / "labels.csv")
            ccol = next(c for c in lab.columns
                        if c.startswith("cluster_") and not c.endswith("_ranked"))
        except Exception:
            continue
        lab["key"] = [f"{p}|{norm(e)}" for p, e in zip(lab["patient_id"], lab["electrode"])]
        m = co["key"].map(lab.dropna(subset=[ccol]).groupby("key")[ccol].apply(
            lambda s: sorted({int(v) for v in s})).to_dict())
        cj["runs"][f"{key.replace('/', '__')}__{run.name}"] = [
            None if not isinstance(v, list) else v for v in m]
    (OUT / "contacts.json").write_text(json.dumps(cj), encoding="utf-8")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cov_mb = sum((OUT / hemis[h]["file"]).stat().st_size for h in hemis) / 1e6
    con_mb = (OUT / "contacts.json").stat().st_size / 1e6
    print(f"\n  {len(entries)} runs | shared {cov_mb:.2f} MB + contacts {con_mb:.2f} MB "
          f"+ per-run {total_mb:.2f} MB = {cov_mb + con_mb + total_mb:.2f} MB")
    print(f"  -> {OUT}")
    return 0


GLASS_VIEWS = ("lateral_L", "lateral_R", "dorsal", "frontal")


def run_figures(run: Path, clusters: list[int], tracked: set[str], k: int) -> dict:
    """Figures the PDF export can pull for this run, as paths relative to the run dir.

    Only files that exist AND are reachable over raw.githubusercontent are listed, so a
    missing figure shrinks the PDF instead of putting a broken box in it.

    Cluster ids come from labels.csv, never from the recon/ directory listing — five runs
    have a different number of recon/cluster_NN dirs than they have clusters (stale
    renders left behind by an earlier K), and enumerating the directory would pair a
    cluster with another run's glass brain.
    """
    def ok(rel: str) -> bool:
        p = run / rel
        if not p.exists():
            return False
        if not tracked:
            return True
        try:
            return str(p.relative_to(ROOT.parent)).replace("\\", "/") in tracked
        except ValueError:
            return False

    per_cluster = {}
    for c in clusters:
        views = {v: f"recon/cluster_{c:02d}/by_condition/{v}.png" for v in GLASS_VIEWS}
        views = {v: rel for v, rel in views.items() if ok(rel)}
        centroid = f"cluster_centroids/cluster_{c:02d}.png"
        entry = {"glass": views}
        if ok(centroid):
            entry["centroid"] = centroid
        if entry["glass"] or "centroid" in entry:
            per_cluster[str(c)] = entry

    overview = []
    for rel, title in (
        ("cluster_cards.png", "Cluster cards — waveform, glass brains, anatomy"),
        (f"timing/timing_k{k:02d}_panel.png", f"Cluster timing panel (K={k})"),
        ("timing/onset_across_k.png", "Onset ordering across K"),
        ("sweep_metrics.png", "K sweep — silhouette, CH, DB, cluster sizes"),
        ("silhouette_by_k.png", "Silhouette by K"),
        ("consensus_heatmap.png", "Consensus heatmap"),
        ("centroids.png", "Cluster centroids"),
        ("centroid_distance_heatmap.png", "Centroid distance matrix"),
    ):
        if ok(rel):
            overview.append({"path": rel, "title": title})

    return {"per_cluster": per_cluster, "overview": overview}


if __name__ == "__main__":
    sys.exit(main())
