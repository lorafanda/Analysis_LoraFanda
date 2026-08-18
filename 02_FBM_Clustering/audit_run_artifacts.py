#!/usr/bin/env python3
"""
audit_run_artifacts.py - what does each clustering run still need?

Every downstream figure on analysis_status.html is built from a per-run artifact,
and the artifacts are produced by four different things (a notebook, a script, or
a notebook that must run before a script). When a run is missing one, the figure
that needs it does not fail loudly - it renders "not generated yet", or the
generator silently falls back to whatever run it was pinned to.

This prints one row per run: what exists, what is missing, and which command
produces the missing piece.

    python audit_run_artifacts.py
    python audit_run_artifacts.py --current-only     # skip superseded runs
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLUST = ROOT / "outputs" / "clustering"

# What each artifact is for, and what makes it.
ARTIFACTS = {
    "labels.csv":        "the run itself (233 / 210-232)",
    "metrics.json":      "the run itself",
    "recon_csv":         "make_recon_csv.py --run <run>",
    "cluster_centroids": "the run itself (fit_and_save)",
    "recon_glassbrains": "252_clustering_recon.ipynb  (cnmf: render_cnmf_glassbrains.py)",
    "cluster_cards":     "make_cluster_cards.py --run <run>",
    "consensus_heatmap": "213_cluster_ranking.ipynb   (211 skips concat_* feature sets)",
    "timing":            "214_concat_timingranking.ipynb",
    "per_cluster_anatomy": "211_validation.ipynb",
}


def probe(rd: Path) -> dict:
    recon = rd / "recon"
    has_glass = recon.is_dir() and any(
        (recon / d.name / "by_condition").is_dir() for d in recon.iterdir() if d.is_dir()
    )
    return {
        "labels.csv": (rd / "labels.csv").exists(),
        "metrics.json": (rd / "metrics.json").exists(),
        "recon_csv": bool(list(recon.glob("*__with_fsaverage.csv"))) if recon.is_dir() else False,
        "cluster_centroids": (rd / "cluster_centroids").is_dir()
                             and any((rd / "cluster_centroids").iterdir()),
        "recon_glassbrains": has_glass,
        "cluster_cards": (rd / "cluster_cards.png").exists(),
        "consensus_heatmap": (rd / "consensus_heatmap.png").exists(),
        "timing": (rd / "timing").is_dir() and any((rd / "timing").iterdir()),
        "per_cluster_anatomy": (rd / "per_cluster_anatomy.csv").exists(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-only", action="store_true")
    a = ap.parse_args()

    idx = json.loads((CLUST / "index.json").read_text(encoding="utf-8"))
    runs = idx["runs"] if isinstance(idx, dict) else idx
    latest = idx.get("latest", {}) if isinstance(idx, dict) else {}

    rows = []
    for r in runs:
        m, f, rid = r["method"], r["feature_set"], r["run_id"]
        rd = CLUST / m / f / "runs" / rid
        if not rd.is_dir():
            rows.append((m, f, rid, None, r, True))
            continue
        is_latest = latest.get(m, {}).get(f) == rid
        if a.current_only and not is_latest:
            continue
        rows.append((m, f, rid, probe(rd), r, is_latest))

    keys = list(ARTIFACTS)
    hdr = f"{'method/feature_set':32s} {'run':17s} {'n':>5s} {'K':>3s} cur "
    hdr += " ".join(k[:9].rjust(9) for k in keys)
    print(hdr)
    print("-" * len(hdr))
    missing_by_art = {k: [] for k in keys}
    for m, f, rid, p, r, is_latest in rows:
        tag = f"{m}/{f}"
        if p is None:
            print(f"{tag:32s} {rid:17s} {'':>5s} {'':>3s}  -  DIRECTORY MISSING")
            continue
        cells = []
        for k in keys:
            cells.append(("   ok    " if p[k] else "  MISS   "))
            if not p[k] and is_latest:
                missing_by_art[k].append(f"{tag} {rid}")
        print(f"{tag:32s} {rid:17s} {r.get('n_samples', ''):>5} "
              f"{r.get('n_clusters', ''):>3} {'*' if is_latest else ' '}  " + " ".join(cells))

    print("\n" + "=" * 78)
    print("MISSING ON CURRENT RUNS ONLY (marked * above) — with the command that fixes it")
    print("=" * 78)
    for k in keys:
        if missing_by_art[k]:
            print(f"\n{k}   <-  {ARTIFACTS[k]}")
            for x in missing_by_art[k]:
                print(f"    {x}")
    if not any(missing_by_art.values()):
        print("  nothing missing on the current runs")
    print(f"\n  audited {len(rows)} runs at {datetime.now():%Y-%m-%d %H:%M}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
