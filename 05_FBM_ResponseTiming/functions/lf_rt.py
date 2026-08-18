"""
lf_rt.py - loaders and timing features for the real-time (GO-aligned) ERSP cubes.

WHAT THIS OPERATES ON
    01_FBM_Analysis/outputs/05_ERSP_LM_RAWONLY_RealTime/<pid>/LM/ERSP_matrix/<cond>/
        <pid>_<cond>_<ref>_ERSP_<chan>_RT_GO.npy      shape (129, T)

    Produced by 150_ERSP_analysis_pipeline_noTwarping.ipynb with
    RT_MODE="RT", RT_ALIGN="go", RT_WINDOW=(-2.5, 5.0), RT_MAX_POST_S=5.0.

THE TWO AXES ARE NOT STORED IN THE .npy. They are reconstructed here, from the
same expressions the pipeline used, and this is the only place that knowledge
lives:

    x = np.linspace(-2.5, 5.0, T, endpoint=False)      lf_ersp.compute_ersp, RT branch
    f = np.linspace(0, 500, 129)                       nfft=256 at fs_ds ~ 1 kHz

    t = 0 is the GO CUE, i.e. stimulus OFFSET. There is no speech-onset event
    anywhere in this dataset, so "response portion" means "after GO", not "after
    the patient started speaking".

T VARIES BY PATIENT AND CONDITION (369-635 across the tree) while the window is
fixed, so the cubes differ in time RESOLUTION, not extent. Everything is
resampled onto one grid before comparison.

THE CUBES ARE TRIAL-AVERAGED. Per-trial data is not persisted anywhere in the 05
tree, so nothing here can correlate a single trial's response time with a single
trial's neural timing. What it can do is compare electrodes, and groups of
electrodes, on the shape of their averaged response. Trial-level work needs the
pipeline to persist per-trial HG first.

VALUES ARE dB RELATIVE TO THE PRE-STIMULUS BASELINE (cfg.baseline_w =
(-0.6, -0.1) s before STIMULUS onset, not before GO). So 0 dB is baseline and
thresholds below are absolute dB. This matters: a fixed pre-GO reference window
would sit in pre-stimulus silence for picture (1 s stimulus) but inside the
stimulus for reading (3.5 s), and would not mean the same thing in the two.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from typing import Iterator

import numpy as np
import pandas as pd

RT_WINDOW = (-2.5, 5.0)
N_FREQ = 129
FMAX = 500.0
HG_BAND = (70.0, 150.0)
CONDITIONS = ("audio", "picture", "reading")

# One grid for everything downstream: 20 ms steps over the full window, which is
# the pipeline's own STFT hop, so this neither invents nor discards resolution.
GRID = np.arange(RT_WINDOW[0], RT_WINDOW[1], 0.02)

_FNAME = re.compile(
    r"^(?P<pid>.+?)_(?P<cond>audio|picture|reading)_(?P<ref>[A-Z]+)_ERSP_"
    r"(?P<chan>.+?)_RT_GO\.npy$")


def normalize_label(s) -> str:
    """'aH_R-1' -> 'AHR1'. Same rule the clustering and recon sides use, so the
    three tables join without a bespoke mapping per source."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def time_axis(n_cols: int, window=RT_WINDOW) -> np.ndarray:
    return np.linspace(window[0], window[1], int(n_cols), endpoint=False)


# ---------------------------------------------------------------------------
# Time-normalised (04) cubes, for the warped-vs-real comparison
# ---------------------------------------------------------------------------
# lf_ersp.compute_ersp, TN branch:  x = np.linspace(0.0, 100.0, N, endpoint=False)
# with cfg.proportions = (0.0, 0.5, 0.5), so 0% is stimulus ONSET and 50% is the
# GO cue. The response portion is 50-100%.
#
# This exists so the same timing features can be measured on the SAME electrodes
# in both time bases. The only difference between the two tables is the axis, so
# any disagreement between them is attributable to the warp and nothing else.
TN_WINDOW = (0.0, 100.0)
TN_GO_PCT = 50.0
TN_GRID = np.arange(0.0, 100.0, 100.0 / 300.0)

_FNAME_TN = re.compile(
    r"^(?P<pid>.+?)_(?P<cond>audio|picture|reading)_(?P<ref>[A-Z]+)_ERSP_"
    r"(?P<chan>.+?)_TN\.npy$")


def tn_time_axis(n_cols: int) -> np.ndarray:
    return np.linspace(TN_WINDOW[0], TN_WINDOW[1], int(n_cols), endpoint=False)


def iter_cubes_tn(tn_root: str, conditions=CONDITIONS):
    """Same walk as iter_cubes but over the time-normalised 04 tree."""
    for pid in sorted(os.listdir(tn_root)):
        base = os.path.join(tn_root, pid, "LM", "ERSP_matrix")
        if not os.path.isdir(base):
            continue
        for cond in conditions:
            d = os.path.join(base, cond)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                m = _FNAME_TN.match(fn)
                if not m:
                    continue
                yield dict(patient_id=m.group("pid"), condition=m.group("cond"),
                           reref=m.group("ref"), electrode=m.group("chan"),
                           contact_norm=normalize_label(m.group("chan")),
                           path=os.path.join(d, fn))


def to_grid_tn(trace: np.ndarray, grid: np.ndarray = TN_GRID) -> np.ndarray:
    x = tn_time_axis(trace.shape[0])
    return np.interp(grid, x, trace, left=np.nan, right=np.nan)


def freq_axis(n_rows: int = N_FREQ, fmax: float = FMAX) -> np.ndarray:
    return np.linspace(0.0, fmax, int(n_rows))


def iter_cubes(rt_root: str, conditions=CONDITIONS) -> Iterator[dict]:
    """Walk the RealTime tree and yield one record per saved cube."""
    for pid in sorted(os.listdir(rt_root)):
        base = os.path.join(rt_root, pid, "LM", "ERSP_matrix")
        if not os.path.isdir(base):
            continue
        for cond in conditions:
            d = os.path.join(base, cond)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                m = _FNAME.match(fn)
                if not m:
                    continue
                yield dict(patient_id=m.group("pid"), condition=m.group("cond"),
                           reref=m.group("ref"), electrode=m.group("chan"),
                           contact_norm=normalize_label(m.group("chan")),
                           path=os.path.join(d, fn))


def band_trace(A: np.ndarray, band=HG_BAND) -> np.ndarray:
    """Mean dB across a frequency band -> one time course.

    Averaging in dB (not power) keeps this identical to what the ERSP figures
    show, so a latency read off the table matches the picture on the page.
    """
    f = freq_axis(A.shape[0])
    sel = (f >= band[0]) & (f <= band[1])
    if not sel.any():
        raise ValueError(f"band {band} outside 0-{FMAX} Hz")
    return np.nanmean(A[sel, :], axis=0)


def to_grid(trace: np.ndarray, grid: np.ndarray = GRID) -> np.ndarray:
    x = time_axis(trace.shape[0])
    return np.interp(grid, x, trace, left=np.nan, right=np.nan)


@dataclass
class Timing:
    """Timing of one electrode's averaged HG response, in seconds after GO."""
    peak_lat: float          # argmax dB in the response window
    peak_db: float
    onset_lat: float         # first sustained crossing of +thr
    offset_lat: float        # last sustained sample above +thr
    dur: float               # offset - onset
    com: float               # centre of mass of the positive part
    auc: float               # integral of max(dB, 0), dB*s
    trough_lat: float        # mirror, for suppressed electrodes
    trough_db: float
    onset_lat_neg: float
    frac_above: float        # fraction of the window above +thr
    frac_below: float        # fraction below -thr
    stim_peak_lat: float     # argmax BEFORE GO, for the stimulus-portion contrast
    stim_peak_db: float


def _sustained(mask: np.ndarray, min_run: int):
    """First and last index of the first run of `mask` at least min_run long.

    A single noisy sample crossing threshold is not an onset; requiring a run is
    what separates a response from a spike in the average.
    """
    if mask.size == 0 or not mask.any():
        return None, None
    idx = np.flatnonzero(mask)
    splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    runs = [s for s in splits if s.size >= min_run]
    if not runs:
        return None, None
    return int(runs[0][0]), int(runs[-1][-1])


def timing_features(trace_on_grid: np.ndarray, *, grid: np.ndarray = GRID,
                    resp=(0.0, 5.0), thr_db: float = 1.0,
                    min_ms: float = 100.0) -> Timing:
    g = grid
    y = np.asarray(trace_on_grid, dtype=float)
    nan = float("nan")
    step = float(np.median(np.diff(g)))
    min_run = max(1, int(round((min_ms / 1000.0) / step)))

    rs = (g >= resp[0]) & (g < resp[1])
    yr, gr = y[rs], g[rs]
    ok = np.isfinite(yr)
    if ok.sum() < min_run:
        return Timing(*([nan] * 14))
    yr = np.where(ok, yr, np.nan)

    i_pk = int(np.nanargmax(yr))
    i_tr = int(np.nanargmin(yr))
    pos = np.nan_to_num(np.clip(yr, 0, None))

    a, b = _sustained(np.nan_to_num(yr, nan=-99) >= thr_db, min_run)
    an, _ = _sustained(np.nan_to_num(yr, nan=+99) <= -thr_db, min_run)

    tot = pos.sum()
    com = float((gr * pos).sum() / tot) if tot > 0 else nan

    pre = (g >= resp[0] - 2.5) & (g < resp[0])
    yp = y[pre]
    if np.isfinite(yp).any():
        i_sp = int(np.nanargmax(yp))
        stim_lat, stim_db = float(g[pre][i_sp]), float(yp[i_sp])
    else:
        stim_lat, stim_db = nan, nan

    return Timing(
        peak_lat=float(gr[i_pk]), peak_db=float(yr[i_pk]),
        onset_lat=float(gr[a]) if a is not None else nan,
        offset_lat=float(gr[b]) if b is not None else nan,
        dur=float(gr[b] - gr[a]) if (a is not None and b is not None) else nan,
        com=com, auc=float(tot * step),
        trough_lat=float(gr[i_tr]), trough_db=float(yr[i_tr]),
        onset_lat_neg=float(gr[an]) if an is not None else nan,
        frac_above=float(np.nanmean(np.nan_to_num(yr, nan=-99) >= thr_db)),
        frac_below=float(np.nanmean(np.nan_to_num(yr, nan=+99) <= -thr_db)),
        stim_peak_lat=stim_lat, stim_peak_db=stim_db,
    )


def timing_row(rec: dict, *, band=HG_BAND, **kw) -> dict:
    A = np.load(rec["path"])
    tr = to_grid(band_trace(A, band))
    out = dict(rec)
    out.pop("path", None)
    out["n_cols"] = int(A.shape[1])
    out.update(asdict(timing_features(tr, **kw)))
    return out


# --------------------------------------------------------------------------
# Group membership: the three ways an electrode is already labelled
# --------------------------------------------------------------------------

def load_clusters(clust_root: str, runs: dict) -> pd.DataFrame:
    """runs: {'kmeans_concat_hg': <run dir>, ...} -> one row per electrode,
    one column per run holding that run's cluster id."""
    out = None
    for name, d in runs.items():
        lab = pd.read_csv(os.path.join(d, "labels.csv"))
        ccol = next(c for c in lab.columns
                    if c.startswith("cluster_") and not c.endswith("_ranked"))
        t = lab[["patient_id", "electrode"]].copy()
        t["contact_norm"] = [normalize_label(e) for e in lab["electrode"]]
        t[name] = lab[ccol].to_numpy()
        t = t.drop(columns=["electrode"])
        out = t if out is None else out.merge(
            t, on=["patient_id", "contact_norm"], how="outer")
    return out


def load_loadings(dec_dir: str) -> pd.DataFrame:
    """Graded convex-NMF weights w0..wK-1 per electrode. Kept separate from the
    hard labels because the whole point of the decomposition is that most
    electrodes have no majority component."""
    p = os.path.join(dec_dir, "electrode_loadings.csv")
    d = pd.read_csv(p)
    w = [c for c in d.columns if re.fullmatch(r"w\d+", c)]
    t = d[["patient_id"] + w].copy()
    t["contact_norm"] = [normalize_label(e) for e in d["electrode"]]
    return t


def load_roles(pool_csv: str) -> pd.DataFrame:
    """Pooling roles. `role` is the single winner, `roles_matched` the full
    semicolon-separated set - an electrode can carry several."""
    d = pd.read_csv(pool_csv)
    t = pd.DataFrame({
        "patient_id": d["patient"].astype(str),
        "contact_norm": [normalize_label(c) for c in d["contact"]],
        "role": d.get("role"),
        "n_roles": d.get("n_roles"),
        "yeo7": d.get("yeo7_network"),
        "is_cortical": d.get("is_cortical"),
    })
    return t.drop_duplicates(["patient_id", "contact_norm"])


# ---------------------------------------------------------------------------
# Statistics shared by every figure script
# ---------------------------------------------------------------------------

def within_patient_perm(df, group, feature, n=2000, seed=0):
    """Shuffle group labels WITHIN each patient; statistic is the spread of group
    medians. Returns (observed_spread, p, n_electrodes, n_groups).

    This is the test, not Kruskal-Wallis. Electrodes inside a patient share a
    brain, a reference, a montage and a response speed, so pooling them and
    calling them independent returns p < 1e-10 for group x condition cells that
    this test scores as null. Shuffling within patient asks the question that
    survives: given this patient's own contacts, does membership still order them
    in time?
    """
    import numpy as _np
    import pandas as _pd
    d = df[["patient_id", group, feature]].dropna()
    if d[group].nunique() < 2 or len(d) < 20:
        return float("nan"), float("nan"), len(d), int(d[group].nunique())

    vals = d[feature].to_numpy()

    def spread(lbl):
        m = _pd.DataFrame({"g": lbl, "v": vals}).groupby("g")["v"].median()
        return float(m.max() - m.min())

    obs = spread(d[group].to_numpy())
    rng = _np.random.default_rng(seed)
    pid = d["patient_id"].to_numpy()
    lbl = d[group].to_numpy().copy()
    idx_by_pat = [_np.flatnonzero(pid == p) for p in _np.unique(pid)]
    null = _np.empty(n)
    for i in range(n):
        sh = lbl.copy()
        for ix in idx_by_pat:
            sh[ix] = rng.permutation(sh[ix])
        null[i] = spread(sh)
    p = float((_np.sum(null >= obs) + 1) / (n + 1))
    return obs, p, len(d), int(d[group].nunique())


def boot_ci(v, n=2000, seed=0):
    """Median with a bootstrap 95% CI. Returns (median, lo, hi)."""
    import numpy as _np
    v = _np.asarray(v, dtype=float)
    v = v[_np.isfinite(v)]
    if v.size < 3:
        return float("nan"), float("nan"), float("nan")
    rng = _np.random.default_rng(seed)
    b = _np.median(rng.choice(v, (n, v.size), replace=True), axis=1)
    return float(_np.median(v)), float(_np.percentile(b, 2.5)), float(_np.percentile(b, 97.5))
