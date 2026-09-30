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

THE COHORT TAG (2026-09-30). Every run manifest written since then carries
"cache": {"name": "concat_source_v<N>", ...} - the dataset cache it was fitted on
(lf_concat stamps the frame, fit_and_save / publish_decomposition write it). The
readers here compare it with the cache that is current now (lf_concat's resolver:
the newest concat_source_v<N> on disk, or LF_CONCAT_CACHE) and REFUSE a mismatch,
with the two names in the message. A run with no tag (fitted before 2026-09-30) is
refused too - "I don't know" is not "current". newest_run() therefore returns the
newest run OF THE CURRENT CACHE, not the newest run: a rebuild on v14 that finds
only v13 fits stops at the first figure and says so, instead of rebuilding the site
on the previous cohort - which is what happened on 2026-09-27 (v12 driver on v11
fits; only FIG 4 refused, because only FIG 4 checked).

To reproduce a published run: LF_CONCAT_CACHE=<path of that cache> makes that
cache current, and its runs pass. To look at a stale run on purpose:
LF_ALLOW_STALE_RUNS=1 turns every refusal into a printed warning.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

CLUST = Path(__file__).resolve().parents[1] / "outputs" / "clustering"


class StaleRunError(RuntimeError):
    """A run fitted on a cache that is not the current one (or on no recorded cache)."""


def current_cache() -> str:
    from functions.lf_concat import current_cache_name       # local: lf_concat is heavier
    return current_cache_name()


def run_cache(run_dir: Path) -> str | None:
    """The cohort tag of a run, or None for a run that predates the tag."""
    man = Path(run_dir) / "manifest.json"
    if not man.exists():
        return None
    try:
        return (json.loads(man.read_text(encoding="utf-8")).get("cache") or {}).get("name")
    except Exception:
        return None


def _allow_stale() -> bool:
    return os.environ.get("LF_ALLOW_STALE_RUNS", "").strip() not in ("", "0", "false", "no")


def check_cache(run_dir: Path, *, what: str = "") -> str:
    """Refuse a run whose cohort tag is not the current cache's. Returns the tag.

    LF_ALLOW_STALE_RUNS=1 downgrades the refusal to a warning (printed once per run).
    """
    rd = Path(run_dir)
    have, want = run_cache(rd), current_cache()
    if have == want:
        return have
    msg = (f"{what + ': ' if what else ''}run {rd.name} ({rd.parent.parent.parent.name}/{rd.parent.parent.name}) "
           + (f"was fitted on {have}" if have else "records no cohort tag (fitted before 2026-09-30)")
           + f", the current cache is {want}. Refit on {want}, or set LF_CONCAT_CACHE to {have or 'that cache'} "
             "to work on that cohort deliberately; LF_ALLOW_STALE_RUNS=1 turns this into a warning.")
    if _allow_stale():
        print(f"[lf_runs] WARNING (stale run allowed): {msg}")
        return have
    raise StaleRunError(msg)


def current_runs(method: str, feature_set: str, *, require_files: tuple = ()) -> list[Path]:
    """Every run of a track that carries the current cohort tag (and the named files),
    newest first. Empty when the track has runs but none of the current cache."""
    d = CLUST / method / feature_set / "runs"
    if not d.is_dir():
        return []
    want = current_cache()
    out = []
    for r in sorted((x for x in d.iterdir() if x.is_dir()), key=lambda x: x.name, reverse=True):
        if all((r / f).exists() for f in require_files) and run_cache(r) == want:
            out.append(r)
    return out


def newest_run(method: str, feature_set: str, *, require_files: tuple = ()) -> Path:
    """Newest run directory of the CURRENT cache for a track (run ids YYYYmmdd_HHMMSS sort).

    Runs of another cache are skipped; if none of the current cache exists the error names
    the newest run there is and what it was fitted on. With LF_ALLOW_STALE_RUNS=1 the
    newest run is returned whatever its tag, with a warning. `require_files` narrows to
    runs that carry those artifacts (e.g. X_train.npy), as several callers need.
    """
    d = CLUST / method / feature_set / "runs"
    if not d.is_dir():
        raise FileNotFoundError(f"no runs directory for {method}/{feature_set}")
    runs = sorted((x for x in d.iterdir() if x.is_dir()), key=lambda x: x.name)
    runs = [r for r in runs if all((r / f).exists() for f in require_files)]
    if not runs:
        raise FileNotFoundError(f"no runs under {d}" + (f" with {', '.join(require_files)}" if require_files else ""))
    cur = current_runs(method, feature_set, require_files=require_files)
    if cur:
        return cur[0]
    # nothing of the current cache: refuse (or warn) on the newest, which says why
    check_cache(runs[-1], what=f"newest run of {method}/{feature_set}")
    return runs[-1]


def resolve_run(method: str, feature_set: str, run: str | None = None) -> Path:
    """`run` may be a full path, a bare run id, or None for the newest of the current cache.
    An explicit run is checked too: naming a stale run by id is the mistake this is for."""
    if run:
        p = Path(run)
        if not p.is_dir():
            p = CLUST / method / feature_set / "runs" / str(run)
        if not p.is_dir():
            raise FileNotFoundError(f"run not found: {run}")
        check_cache(p, what=f"{method}/{feature_set}")
        return p
    return newest_run(method, feature_set)


def run_info(run_dir: Path) -> dict:
    """Everything needed to stamp a figure, read from the run's own files."""
    rd = Path(run_dir)
    info = {"run_id": rd.name, "path": str(rd), "n_electrodes": None,
            "n_patients": None, "k": None, "method": None, "feature_set": None,
            "cache": run_cache(rd)}
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
    if i.get("cache"):
        bits.append(i["cache"])                    # the cohort tag on the figure itself
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
