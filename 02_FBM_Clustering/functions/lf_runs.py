"""
lf_runs.py - resolve which clustering run a figure script should use, and say so.

Three scripts used to hard-code `.../kmeans/concat_hg/runs/20260803_175417`. When
the cohort moved from 1027 electrodes / 24 patients to 1266 / 27, re-running them
silently reproduced the OLD cohort - the figure came out looking fresh, dated
today, and describing a superseded electrode set. A fourth script had the mirror
problem: it auto-selected the newest run and would render panels for artifacts
that run did not have yet.

So: resolve explicitly, and stamp what was resolved onto the figure. A figure that
carries its run id, cohort size and date cannot quietly become stale - you can
read the provenance off the image itself.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

CLUST = Path(__file__).resolve().parents[1] / "outputs" / "clustering"


def newest_run(method: str, feature_set: str) -> Path:
    """Newest run directory for a track. Run ids are YYYYmmdd_HHMMSS so they sort."""
    d = CLUST / method / feature_set / "runs"
    if not d.is_dir():
        raise FileNotFoundError(f"no runs directory for {method}/{feature_set}")
    runs = sorted((x for x in d.iterdir() if x.is_dir()), key=lambda x: x.name)
    if not runs:
        raise FileNotFoundError(f"no runs under {d}")
    return runs[-1]


def resolve_run(method: str, feature_set: str, run: str | None = None) -> Path:
    """`run` may be a full path, a bare run id, or None for the newest."""
    if run:
        p = Path(run)
        if p.is_dir():
            return p
        p = CLUST / method / feature_set / "runs" / str(run)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"run not found: {run}")
    return newest_run(method, feature_set)


def run_info(run_dir: Path) -> dict:
    """Everything needed to stamp a figure, read from the run's own files."""
    rd = Path(run_dir)
    info = {"run_id": rd.name, "path": str(rd), "n_electrodes": None,
            "n_patients": None, "k": None, "method": None, "feature_set": None}
    try:
        parts = rd.relative_to(CLUST).parts
        info["method"], info["feature_set"] = parts[0], parts[1]
    except Exception:
        pass
    man = rd / "manifest.json"
    if man.exists():
        try:
            m = json.loads(man.read_text(encoding="utf-8"))
            s = m.get("summary", {})
            info["n_electrodes"] = s.get("n_samples")
            info["k"] = s.get("n_clusters") or m.get("params", {}).get("k")
        except Exception:
            pass
    met = rd / "metrics.json"
    if info["n_electrodes"] is None and met.exists():
        try:
            m = json.loads(met.read_text(encoding="utf-8"))
            info["n_electrodes"] = m.get("n_samples")
            info["k"] = m.get("n_clusters")
        except Exception:
            pass
    lab = rd / "labels.csv"
    if lab.exists():
        try:
            import pandas as pd
            d = pd.read_csv(lab, usecols=lambda c: c in ("patient_id",))
            info["n_patients"] = int(d["patient_id"].nunique())
            if info["n_electrodes"] is None:
                info["n_electrodes"] = int(len(d))
        except Exception:
            pass
    return info


def provenance(run_dir: Path, extra: str = "") -> str:
    """One line for a figure footer / suptitle. Carries the cohort AND the date,
    so a stale figure is identifiable without opening the run directory."""
    i = run_info(run_dir)
    bits = [f"{i['method']}/{i['feature_set']}" if i["method"] else Path(run_dir).name,
            f"run {i['run_id']}"]
    if i["n_electrodes"] is not None and i["n_patients"] is not None:
        bits.append(f"{i['n_electrodes']} electrodes / {i['n_patients']} patients")
    elif i["n_electrodes"] is not None:
        bits.append(f"{i['n_electrodes']} electrodes")
    if i["k"]:
        bits.append(f"K={i['k']}")
    bits.append(f"rendered {datetime.now():%Y-%m-%d}")
    if extra:
        bits.append(extra)
    return "  ·  ".join(bits)


def require(run_dir: Path, *artifacts: str) -> None:
    """Fail loudly BEFORE rendering if a needed artifact is absent.

    The alternative - which is what happened - is a figure with a hole in it and
    no error, which then gets published.
    """
    rd = Path(run_dir)
    missing = []
    for a in artifacts:
        p = rd / a
        ok = p.exists() and (not p.is_dir() or any(p.iterdir()))
        if not ok:
            missing.append(a)
    if missing:
        raise FileNotFoundError(
            f"{rd.name} is missing {', '.join(missing)} — generate those first "
            f"(see audit_run_artifacts.py) rather than rendering a figure with holes")
