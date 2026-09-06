#!/usr/bin/env python3
"""
make_coverage_bundle.py — per-vertex sampling counts for clustering_visualizer.html.

The question the page answers: at this point on the cortex, HOW WELL DID WE SAMPLE, and
of the electrodes that are here, what fraction belong to a given cluster? Every brain
figure in this project carries the caveat "dot density reflects where electrodes were
implanted, not where language is" — this makes that checkable instead of a footnote.

Why not reuse the activity-visualizer projection: that one keeps the k=4 NEAREST
contacts per vertex, so a patient count taken from it can never exceed 4 of 35 and is
useless as a probability. Here every contact within the radius of a vertex is counted,
which is what makes "fraction of patients sampled here" meaningful.

LAYOUT — the counts split into two kinds, because they have different lifetimes:

    coverage_r{R}_{lh,rh}_u8.bin          n_contacts, n_patients    SHARED by every run
    runs/<id>/clusters_r{R}_{h}_u8.bin    n_cluster_0 .. n_cluster_{K-1}    per run

Sampling depends only on where the electrodes are, so it is identical across runs and
is stored once. Only the cluster breakdown is per-run. That is what makes shipping many
runs affordable (~1.6 MB per run per radius instead of ~4.6 MB).

Everything is emitted at several radii (RADII_MM). The radius is a boxcar smoothing
kernel, not a neutral display setting: it decides how much cortex has an estimate at
all and how much evidence each estimate rests on. Shipping one radius would bake that
choice in and hide it, so the page exposes it and the reader can watch a map hold or
dissolve as it changes.

uint8 is checked, not assumed: if any count exceeds 255 the file is written as uint16
and the manifest says so, so a larger radius cannot silently wrap around.

Per vertex this gives the page everything it needs to compute both maps as plain ratios:

    coverage   P(sampled)          = n_patients / n_patients_total
    cluster k  P(cluster k | here) = n_cluster_k / (labelled contacts here)

Counts, not probabilities, are shipped on purpose: the ratio you want depends on the
question, and a stored ratio hides its own denominator.

    python make_coverage_bundle.py                    # every run, every radius
    python make_coverage_bundle.py --radii 8 12
    python make_coverage_bundle.py --run <run dir>    # just one
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
# The cuts the K control offers. Narrower than what a run's sweep holds (4..23) on
# purpose: 5..12 spans every K anyone has argued for here, and each extra cut is a
# labels vector per run to ship and a stability fit to compute.
SWEEP_KS = tuple(range(5, 13))
PART_BYTES = 4 * 1024 * 1024   # keep every emitted .bin under the 5 MB commit guard
RADII_MM = (6.0, 10.0, 15.0)   # the radius is a smoothing kernel; the page lets you vary it
DEFAULT_RADIUS = 10.0

# Tracks offered in the page's dropdown, in display order. Only the NEWEST run per track
# is bundled by default (--per-track raises it): within a track the older runs are
# superseded predecessors on smaller cohorts, and mixing cohorts silently is the exact
# problem the 1027-electrode cleanup fixed.
#
# Deliberately absent: every *_raw track (~6 GB, no consensus/ranking/stability, three of
# them pinned at K=20 = the sweep ceiling) and hierarchical/concat_rawds (K=20 for the
# same reason, and the largest payload of the set).
# The third field is the cohort IDENTITY, not its display label. It is the grouping
# key, so it has to be stable - and it must not carry an electrode count, because the
# count is a property of whatever cohort was last built. Every gated concat run said
# "1266-electrode concat cohort" while holding 1693. The count is composed back in
# after grouping, from the runs.
TRACKS = [
    ("cnmf/concat_hg", "Convex NMF - graded (argmax)", "gated concat cohort"),
    ("cnmf/concat_rawds", "Convex NMF 15-band - graded (argmax)", "gated concat cohort"),
    ("kmeans/concat_hg", "K-means · concatenated", "gated concat cohort"),
    ("hierarchical/concat_hg", "Ward · concatenated", "gated concat cohort"),
    ("kmeans/concat_rawds", "K-means · concat 15-band", "gated concat cohort"),
    ("hierarchical/concat_rawds", "Ward · concat 15-band", "gated concat cohort"),
    # THE 5-BAND SETS. A track missing from this list is invisible to the visualizer
    # however complete its runs are - which is exactly what happened to concat_bands5
    # and concat_bands5z: fitted on all three methods and unreachable from the page.
    ("cnmf/concat_bands5", "Convex NMF 5-band - graded (argmax)", "gated concat cohort"),
    ("cnmf/concat_bands5z", "Convex NMF 5-band z - graded (argmax)", "gated concat cohort"),
    ("kmeans/concat_bands5", "K-means · concat 5-band", "gated concat cohort"),
    ("kmeans/concat_bands5z", "K-means · concat 5-band, z-scored", "gated concat cohort"),
    ("hierarchical/concat_bands5", "Ward · concat 5-band", "gated concat cohort"),
    ("hierarchical/concat_bands5z", "Ward · concat 5-band, z-scored", "gated concat cohort"),
    # archetypal analysis, the fourth method. Listed for all four feature sets; a track
    # with no runs yet is skipped, so these populate as 243 produces them.
    # archetypal analysis dropped 2026-09-06: its runs, outputs and bundle entries are gone
    ("cnmf/concat_hg_all", "Convex NMF - ungated (argmax)", "ungated concat cohort"),
    ("kmeans/concat_hg_all", "K-means · concatenated, ungated", "ungated concat cohort"),
    ("hierarchical/concat_hg_all", "Ward · concatenated, ungated", "ungated concat cohort"),
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


# Patient colours copied from the MOBA palette (lorafanda.github.io/colors_config.json)
# so one patient is the same colour in both tools. MOBA indexes by the patient's
# alphabetical position WITHIN its id-prefix group, which is reproduced here.
PALETTE = {
    "EL": [
        "#0000ff",
        "#00bfff",
        "#00ffff",
        "#008080",
        "#008000",
        "#00ff00",
        "#b0e0e6",
        "#228b22",
        "#e0ffff",
        "#556b2f"
    ],
    "PAT": [
        "#8a2be2",
        "#ff00ff",
        "#ff1493",
        "#dc143c",
        "#ffc0cb",
        "#ff0000",
        "#d2691e",
        "#ffd700",
        "#800080",
        "#8b4513",
        "#fffacd",
        "#fff0f5",
        "#00ff00",
        "#b0e0e6",
        "#228b22",
        "#e0ffff"
    ],
    "MICRO": [
        "#000080",
        "#2f4f4f",
        "#000000",
        "#8b0000",
        "#556b2f"
    ]
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


def write_neighbour_index(verts_by_hemi, xyz, hemi, radii, out: Path) -> dict:
    """Which contacts lie within R mm of each vertex, as CSR - and nothing about K.

    The per-run clusters_*.bin files are K-shaped: K bytes per vertex, so K=6 costs
    5.9 MB and K=23 costs 22.6 MB, per run. That is affordable while a run publishes one
    K and impossible the moment a K control exists - K=5..12 across six runs would be
    roughly 380 MB.

    This is the same neighbourhood, stored once. It is pure geometry: it does not depend
    on K, on the clustering, or on which run is being viewed, so ONE copy serves every
    cut of every run, now and later. Measured on this cohort it is 44 MB for all three
    radii against 116 MB of per-run counts it can replace.

    Layout, per radius per hemisphere:
        nbr_r{R}_{h}_off.bin   uint32, nverts+1   CSR row offsets
        nbr_r{R}_{h}_idx.bin   uint16, nnz        GLOBAL contact indices

    Global rather than per-hemisphere indices so a reader never needs a second mapping
    table; there are 5247 contacts, comfortably inside uint16.
    """
    from scipy.spatial import cKDTree
    out.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=float)
    hemi = np.asarray(hemi)
    if len(xyz) > 65535:
        raise SystemExit(f"  !! {len(xyz)} contacts exceeds uint16 - widen nbr idx dtype")

    index = {}
    for h in ("lh", "rh"):
        gidx = np.where(hemi == h)[0].astype(np.uint16)
        tree = cKDTree(xyz[gidx])
        V = verts_by_hemi[h]
        for r in radii:
            nb = tree.query_ball_point(V, r=r)
            lens = np.fromiter((len(x) for x in nb), dtype=np.int64, count=len(nb))
            off = np.zeros(len(nb) + 1, dtype=np.uint32)
            off[1:] = np.cumsum(lens).astype(np.uint32)
            flat = (np.concatenate([np.asarray(x, dtype=np.int64) for x in nb])
                    if off[-1] else np.zeros(0, dtype=np.int64))
            idx = gidx[flat] if len(flat) else np.zeros(0, dtype=np.uint16)
            fo = out / f"nbr_r{r:g}_{h}_off.bin"
            off.tofile(fo)
            # SPLIT. r=15 lh is 7.6M pairs = 15 MB in one file, and .githooks/pre-commit
            # refuses anything over 5 MB - a guard worth keeping rather than bypassing,
            # since it is the only thing standing between this repo and render bloat.
            # Parts concatenate back to one Uint16Array in the reader.
            iu = idx.astype(np.uint16)
            per = PART_BYTES // 2
            parts = []
            n_parts = max(1, -(-len(iu) // per))
            for j in range(n_parts):
                fp = out / f"nbr_r{r:g}_{h}_idx.p{j:02d}.bin"
                iu[j * per:(j + 1) * per].tofile(fp)
                parts.append(fp.name)
            index.setdefault(f"{r:g}", {})[h] = {
                "off": fo.name, "idx_parts": parts,
                "n_vertices": int(len(nb)), "nnz": int(off[-1]),
            }
            mb = (fo.stat().st_size + sum((out / q).stat().st_size for q in parts)) / 1048576
            print(f"    index r{r:g} {h}: {int(off[-1]):,} pairs, {mb:.1f} MB "
                  f"in {len(parts)} part(s)")
    return index


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
    ap.add_argument("--radii", type=float, nargs="+", default=list(RADII_MM))
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

    co_all = pd.read_csv(COORDS).dropna(subset=["x", "y", "z"])
    co_all["key"] = [f"{p}|{norm(n)}" for p, n in zip(co_all["patient"], co_all["name"])]
    co_all["hemi1"] = co_all["hemi"].astype(str).str.upper().str[0]

    # ── read every run's labels first, because the PATIENT SET is what defines a cohort
    #
    # The coordinate file carries 35 patients, but only the ones that actually entered a
    # run are part of FBM. Counting the other 9 inflated both the numerator and the
    # denominator of every coverage map. Coverage is therefore computed per cohort, over
    # that cohort's contacts only, against that cohort's patient count.
    loaded = []
    for key, label, cohort_name, run in runs:
        try:
            lab = pd.read_csv(run / "labels.csv")
            ccol = next(c for c in lab.columns
                        if c.startswith("cluster_") and not c.endswith("_ranked"))
        except Exception as e:                                   # noqa: BLE001
            print(f"  !! skip {key}/{run.name}: {e}")
            continue
        clusters = sorted(int(c) for c in lab[ccol].dropna().unique())
        if not clusters:
            print(f"  !! skip {key}/{run.name}: no cluster labels")
            continue
        lab["key"] = [f"{p}|{norm(e)}" for p, e in zip(lab["patient_id"], lab["electrode"])]
        pats = frozenset(str(p) for p in lab["patient_id"].dropna().unique())
        loaded.append(dict(key=key, label=label, cohort_name=cohort_name, run=run,
                           lab=lab, ccol=ccol, clusters=clusters, pats=pats))

    if not loaded:
        print("  no usable runs", file=sys.stderr)
        return 1

    # Group runs that share a patient set AND a declared cohort name. The patient set
    # alone was enough while every cohort differed by which patients were in it, and it is
    # not enough any more: concat_hg_all has the SAME 27 patients as the gated concat
    # cohort but 2946 electrodes instead of 1266, so grouping on patients alone merged
    # them and every ungated run was labelled "1266-electrode concat cohort".
    #
    # The coverage NUMBERS were not affected, and it is worth being precise about why:
    # coverage is computed from the cohort patients' contacts in the coordinate file, not
    # from the electrode subset a run happens to cluster, so both cohorts legitimately
    # give 1748 lh / 1863 rh. What was wrong was the identity and the label the report
    # prints - a 2946-electrode run describing itself as the 1266-electrode cohort.
    cohorts, order = {}, []
    for L in loaded:
        cid = next((c for c in order if cohorts[c]["pats"] == L["pats"]
                    and cohorts[c]["identity"] == L["cohort_name"]), None)
        if cid is None:
            cid = f"cohort{len(order) + 1}_n{len(L['pats'])}"
            cohorts[cid] = {"pats": L["pats"], "identity": L["cohort_name"],
                            "label": L["cohort_name"], "runs": []}
            order.append(cid)
        cohorts[cid]["runs"].append(L)
        L["cohort_id"] = cid

    # THE DISPLAYED LABEL IS COMPOSED HERE, from the runs that ended up in the cohort,
    # so it cannot describe a cohort size that no longer exists. Runs in one cohort share
    # an electrode set; if they somehow do not, say so rather than picking one count and
    # presenting it as the cohort's.
    for cid in order:
        c = cohorts[cid]
        # len(lab), not L["n_electrodes"]: that key is only added later, when the
        # manifest entry is built, and this runs before it
        ns = sorted({int(len(L["lab"])) for L in c["runs"] if L.get("lab") is not None})
        if len(ns) == 1:
            c["label"] = f"{ns[0]}-electrode {c['identity']}"
        elif ns:
            c["label"] = (f"{c['identity']} ({min(ns)}-{max(ns)} electrodes, "
                          f"NOT one electrode set)")
            print(f"  !! {cid}: runs disagree on electrode count {ns} - they should not "
                  f"share a cohort", file=sys.stderr)
        else:
            c["label"] = c["identity"]

    print(f"  {len(loaded)} run(s), {len(order)} cohort(s), radii {[f'{r:g}' for r in a.radii]}")
    for cid in order:
        c = cohorts[cid]
        print(f"    {cid}: {len(c['pats'])} patients, {len(c['runs'])} run(s) — {c['label']}")

    nverts = {}
    verts_by_hemi = {}
    for h in ("lh", "rh"):
        v = nib.load(str(MESHES / f"fsaverage_{h}.gii")).darrays[0].data.astype(float)
        verts_by_hemi[h] = v
        nverts[h] = int(len(v))

    manifest_cohorts = {}
    contacts_json = {"xyz": [], "hemi": [], "patient": [], "cohorts": {}, "runs": {}}
    # one global contact table; each cohort names the indices it owns
    co_all = co_all.reset_index(drop=True)
    contacts_json["xyz"] = [[round(float(r.x), 1), round(float(r.y), 1), round(float(r.z), 1)]
                            for r in co_all.itertuples()]
    contacts_json["hemi"] = ["lh" if r == "L" else "rh" for r in co_all["hemi1"]]
    contacts_json["patient"] = [str(p) for p in co_all["patient"]]

    # ── K-independent neighbourhood index, written once for every run and every K
    print("  neighbourhood index (shared by every run and every K):")
    # FULL-PRECISION coordinates, not contacts_json["xyz"]. That one is rounded to
    # 0.1 mm for transport, and rounding moves contacts across the radius boundary: built
    # from it, the index disagreed with the counts by up to 27 at r=15.
    nbr_index = write_neighbour_index(
        verts_by_hemi, co_all[["x", "y", "z"]].to_numpy(float),
        np.asarray(["lh" if t == "L" else "rh" for t in co_all["hemi1"]]),
        a.radii, OUT)

    total_mb = 0.0
    for cid in order:
        C = cohorts[cid]
        sel = co_all[co_all["patient"].astype(str).isin(C["pats"])].reset_index()
        contacts_json["cohorts"][cid] = [int(i) for i in sel["index"]]
        pat_codes = {p: i for i, p in enumerate(sorted(sel["patient"].astype(str).unique()))}
        n_pat = len(C["pats"])

        near, subs, cov_entry = {}, {}, {}
        for h, tag in (("lh", "L"), ("rh", "R")):
            sub = sel[sel["hemi1"] == tag].reset_index(drop=True)
            subs[h] = sub
            tree = cKDTree(sub[["x", "y", "z"]].to_numpy())
            pc = sub["patient"].astype(str).map(pat_codes).to_numpy()
            near[h] = {}
            for r in a.radii:
                nb = tree.query_ball_point(verts_by_hemi[h], r=r)
                near[h][r] = nb
                cov = np.zeros((nverts[h], 2), dtype=np.int32)
                for v, hits in enumerate(nb):
                    if hits:
                        cov[v, 0] = len(hits)
                        cov[v, 1] = len(set(pc[hits]))
                fn, dt = write_counts(cov, OUT / f"coverage_{cid}_r{r:g}_{h}")
                pct = 100 * float((cov[:, 1] > 0).mean())
                cov_entry.setdefault(f"{r:g}", {})[h] = {
                    "file": fn, "dtype": dt,
                    "pct_vertices_sampled": round(pct, 1),
                    "max_patients_at_a_vertex": int(cov[:, 1].max()),
                    "max_contacts_at_a_vertex": int(cov[:, 0].max())}
                total_mb += (OUT / fn).stat().st_size / 1e6
            print(f"    {cid} {h}: {len(sub)} contacts | " +
                  " ".join(f"r{r:g}={cov_entry[f'{r:g}'][h]['pct_vertices_sampled']:.0f}%"
                           for r in a.radii))

        manifest_cohorts[cid] = {
            "label": C["label"],
            "patients": sorted(C["pats"]),
            "n_patients": n_pat,
            "n_contacts": int(len(sel)),
            "coverage": cov_entry,
        }

        # ── per-run cluster counts, on this cohort's neighbourhoods
        for L in C["runs"]:
            lab, ccol, clusters = L["lab"], L["ccol"], L["clusters"]
            cidx = {k: i for i, k in enumerate(clusters)}
            # one electrode can carry several labels (per-condition runs); every row counts
            clu_of = lab.dropna(subset=[ccol]).groupby("key")[ccol].apply(
                lambda s: [int(v) for v in s]).to_dict()
            multi = int((lab.groupby("key")[ccol].nunique() > 1).sum())
            unit = "electrode" if int(lab.groupby("key")[ccol].size().max()) == 1 \
                else "electrode x condition"

            rdir = OUT / "runs" / f"{L['key'].replace('/', '__')}__{L['run'].name}"
            rdir.mkdir(parents=True, exist_ok=True)
            files, mb = {}, 0.0
            matched = n_samples = 0
            for h, tag in (("lh", "L"), ("rh", "R")):
                lists = subs[h]["key"].map(clu_of).to_numpy()
                matched += int(sum(1 for x in lists if isinstance(x, list)))
                n_samples += int(sum(len(x) for x in lists if isinstance(x, list)))
                for r in a.radii:
                    arr = np.zeros((nverts[h], len(clusters)), dtype=np.int32)
                    for v, hits in enumerate(near[h][r]):
                        for i in hits:
                            lb = lists[i]
                            if isinstance(lb, list):
                                for c in lb:
                                    arr[v, cidx[c]] += 1
                    fn, dt = write_counts(arr, rdir / f"clusters_r{r:g}_{h}")
                    files.setdefault(f"{r:g}", {})[h] = {"file": fn, "dtype": dt}
                    mb += (rdir / fn).stat().st_size / 1e6
            total_mb += mb

            # patient composition, for the pie charts: how many samples each patient
            # contributed to each cluster, and to the run overall
            comp = {}
            for k in clusters:
                vc = lab.loc[lab[ccol] == k, "patient_id"].astype(str).value_counts()
                comp[str(k)] = {p: int(n) for p, n in vc.items()}
            run_pat_totals = {p: int(n) for p, n in
                              lab["patient_id"].astype(str).value_counts().items()}

            L["entry"] = {
                "id": f"{L['key'].replace('/', '__')}__{L['run'].name}",
                "track": L["key"], "track_label": L["label"],
                "cohort": C["label"], "cohort_id": cid,
                "run": L["run"].name,
                "path": f"{L['key']}/runs/{L['run'].name}",
                "dir": f"runs/{L['key'].replace('/', '__')}__{L['run'].name}",
                "k": len(clusters), "clusters": clusters,
                "n_electrodes": int(len(lab)),
                "n_contacts_with_cluster": matched,
                "n_samples_with_cluster": n_samples,
                "unit": unit,
                "n_electrodes_multilabel": multi,
                "n_patients_run": int(lab["patient_id"].nunique()),
                "cluster_sizes": {str(k): int((lab[ccol] == k).sum()) for k in clusters},
                "cluster_patients": {str(k): int(lab.loc[lab[ccol] == k, "patient_id"].nunique())
                                     for k in clusters},
                "cluster_patient_counts": comp,
                "patient_totals": run_pat_totals,
                "counts": files,
                "figures": run_figures(L["run"], clusters, tracked, len(clusters)),
            }
            e = L["entry"]
            print(f"      {L['key']}/{L['run'].name}: K={e['k']}, {e['n_electrodes']} rows "
                  f"[{unit}], {matched} contacts / {n_samples} samples"
                  f"{f', {multi} multi-label' if multi else ''} | {mb:.1f} MB")

    # per-run label vector over the GLOBAL contact table, for the electrode overlay
    contacts_json["loadings"] = {}
    for L in loaded:
        m = co_all["key"].map(L["lab"].dropna(subset=[L["ccol"]]).groupby("key")[L["ccol"]]
                              .apply(lambda s: sorted({int(v) for v in s})).to_dict())
        contacts_json["runs"][L["entry"]["id"]] = [
            None if not isinstance(v, list) else v for v in m]

        # PER-K LABELS. Same vector as above, once per cut in the run's own sweep, so
        # the K control has something to switch to without refitting anything. Written
        # beside the run's counts rather than into contacts.json: it is only needed once
        # a run is selected, and contacts.json is fetched on every page load.
        lbk = L["run"] / "cluster_labels_by_k.csv"
        if lbk.exists():
            sw = pd.read_csv(lbk)
            if len(sw) == len(L["lab"]):
                keys = L["lab"]["key"].to_numpy()
                payload, ks_have = {}, []
                for c in sw.columns:
                    if not c.startswith("k_") or not c[2:].isdigit():
                        continue
                    kk = int(c[2:])
                    if kk not in SWEEP_KS:
                        continue
                    # a LIST, duplicates kept. On an electrode x condition run the same
                    # electrode can land in one cluster under two conditions, and the
                    # counts those files carry are per sample, not per electrode.
                    # contacts.json["runs"] dedupes to a set, which is why rebuilding a
                    # multi-label run from it undercounts; per-K labels must not repeat
                    # that.
                    per = {}
                    for key, v in zip(keys, sw[c].to_numpy()):
                        per.setdefault(key, []).append(int(v))
                    mk = co_all["key"].map({q: sorted(w) for q, w in per.items()})
                    payload[str(kk)] = [None if not isinstance(v, list) else v for v in mk]
                    ks_have.append(kk)
                # Cluster sizes and patient composition PER K, computed the same way the
                # published entry computes its own - over the run's rows, not over the
                # localised contacts. Recomputing these in the browser from
                # labels_by_k.json would silently switch denominator: this run has 1266
                # rows but only 1035 contacts with coordinates, so the sizes would not
                # sum to the run and would disagree with every published figure.
                stats = {}
                pid = L["lab"]["patient_id"].astype(str)
                for c in sw.columns:
                    if not c.startswith("k_") or not c[2:].isdigit():
                        continue
                    kk = int(c[2:])
                    if kk not in SWEEP_KS:
                        continue
                    col = sw[c].to_numpy()
                    ids = sorted({int(v) for v in col})
                    comp_k = {}
                    for j in ids:
                        vc = pid[col == j].value_counts()
                        comp_k[str(j)] = {q: int(n) for q, n in vc.items()}
                    stats[str(kk)] = {
                        "clusters": ids,
                        "cluster_sizes": {str(j): int((col == j).sum()) for j in ids},
                        "cluster_patients": {str(j): int(pid[col == j].nunique())
                                             for j in ids},
                        "cluster_patient_counts": comp_k,
                    }

                # PER-K GRADED WEIGHTS, for the runs that have them. The published
                # w0..wK-1 columns describe one cut; driving the loading gate and the
                # size/grey ramp with them at another K would size electrodes by a
                # partition that is not on screen. sweep_decomposition writes
                # loadings_by_k/G_kNN.npy, already row-normalised.
                loads = {}
                gdir = L["run"] / "loadings_by_k"
                if gdir.is_dir():
                    for kk in sorted(payload):
                        gp = gdir / f"G_k{int(kk):02d}.npy"
                        if not gp.exists():
                            continue
                        G = np.load(gp)
                        if len(G) != len(L["lab"]):
                            print(f"      !! {L['run'].name}: G_k{int(kk):02d} has {len(G)} "
                                  f"rows vs {len(L['lab'])} labels - skipped")
                            continue
                        topk = pd.Series(G.max(axis=1), index=L["lab"].index)
                        dk = (L["lab"].assign(_t=topk).dropna(subset=["_t"])
                              .groupby("key")["_t"].max().to_dict())
                        vk = co_all["key"].map(dk)
                        loads[kk] = [None if v != v else round(float(v), 3) for v in vk]

                if payload:
                    rdir = OUT / "runs" / f"{L['key'].replace('/', '__')}__{L['run'].name}"
                    rdir.mkdir(parents=True, exist_ok=True)
                    (rdir / "labels_by_k.json").write_text(json.dumps(payload),
                                                           encoding="utf-8")
                    (rdir / "sweep_stats.json").write_text(json.dumps(stats),
                                                           encoding="utf-8")
                    L["entry"]["sweep"] = {"ks": sorted(ks_have),
                                           "labels": "labels_by_k.json",
                                           "stats": "sweep_stats.json"}
                    if loads:
                        (rdir / "loadings_by_k.json").write_text(json.dumps(loads),
                                                                 encoding="utf-8")
                        L["entry"]["sweep"]["loadings"] = "loadings_by_k.json"
            else:
                print(f"      !! {L['run'].name}: sweep has {len(sw)} rows vs "
                      f"{len(L['lab'])} labels - per-K labels skipped")

        # GRADED MEMBERSHIP, where the run has it. Only the convex-NMF runs carry
        # w0..wK-1; k-means and Ward are hard partitions where every contact's
        # membership is 1.0 by construction, so exporting a loading for them would
        # invite a control that looks live and cannot mean anything.
        wcols = sorted((c for c in L["lab"].columns
                        if c.startswith("w") and c[1:].isdigit()),
                       key=lambda c: int(c[1:]))
        if not wcols:
            continue
        top = L["lab"][wcols].max(axis=1)
        d = (L["lab"].assign(_top=top).dropna(subset=["_top"])
             .groupby("key")["_top"].max().to_dict())
        vals = co_all["key"].map(d)
        contacts_json["loadings"][L["entry"]["id"]] = [
            None if v != v else round(float(v), 3) for v in vals]
        L["entry"]["has_loadings"] = True

    manifest = {
        "radii_mm": [f"{r:g}" for r in a.radii],
        "default_radius": f"{DEFAULT_RADIUS:g}",
        "coords": COORDS.name,
        "n_contacts_in_coords": int(len(co_all)),
        "n_patients_in_coords": int(co_all["patient"].nunique()),
        "coverage_fields": ["n_contacts", "n_patients"],
        "hemis": {h: {"nvert": nverts[h]} for h in ("lh", "rh")},
        "cohorts": manifest_cohorts,
        "contacts_file": "contacts.json",
        # geometry only: which contacts are within R mm of each vertex, CSR, global
        # contact indices. Lets a reader build the counts for ANY labelling, which is
        # what makes a K control possible without shipping counts per K.
        "neighbours": nbr_index,
        "sweep_ks": list(SWEEP_KS),
        "runs": [L["entry"] for L in loaded],
        "patient_palette": PALETTE,
        "note": ("counts within radius_mm of each vertex, restricted to the patients of "
                 "the run's own cohort; ratios are computed client-side so the "
                 "denominator stays visible"),
    }
    (OUT / "contacts.json").write_text(json.dumps(contacts_json), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    con_mb = (OUT / "contacts.json").stat().st_size / 1e6
    print(f"\n  {len(loaded)} runs, {len(order)} cohorts | counts {total_mb:.1f} MB "
          f"+ contacts {con_mb:.2f} MB")
    print(f"  patients: {manifest['n_patients_in_coords']} in the coordinate file, "
          f"{sorted({len(c['pats']) for c in cohorts.values()})} per FBM cohort")
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
        ("consensus_heatmap.png", "Consensus heatmap"),
        ("centroids.png", "Cluster centroids"),
        ("centroid_distance_heatmap.png", "Centroid distance matrix"),
    ):
        if ok(rel):
            overview.append({"path": rel, "title": title})

    return {"per_cluster": per_cluster, "overview": overview}


if __name__ == "__main__":
    sys.exit(main())
