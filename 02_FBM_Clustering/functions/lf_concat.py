"""
lf_concat.py — CONDITION-CONCATENATED clustering representations.

Every other clustering track (210 raw / 212 rawds / 231 minus101 / 232 hg) treats
one sample = one **electrode x condition**, so the strongest structure available to
find is the condition/modality axis. This module builds the complementary view:

    one sample = one ELECTRODE, features = [audio | picture | reading] stitched in time

which asks "what functional TYPE is this contact across the whole task?" instead of
"which condition is this response?". It is the same sample construction the stage-03
parcellation decoding and the stage-04 pooling already use, so clusters from this
track are directly comparable to those results.

Sample filter (mirrors 03's parcellation task):
    keep an electrode iff it has ALL THREE conditions present, and (by default) is
    high-activity in >= 1 of them. The per-condition high-activity gate is computed
    the usual way but applied at the ELECTRODE level, so a contact that only responds
    in one condition still contributes its full three-condition profile.

Three feature sets, mirroring the per-condition tracks:
    concat_hg     1 x 900   HG line (70-150 Hz mean) per condition, stitched
    concat_rawds  15 x 90   15 canonical bands x 30 time bins per condition (the
                            SAME grid stage-04 pooling matches roles on)
    concat_bands5 5 x 90    the SAME grid coarsened to 5 bands, each a union of
                            contiguous 15-band edges. Averages the ORIGINAL frequency
                            bins, so it is bandwidth-weighted rather than a mean of
                            pre-averaged bands - and it is the same builder with
                            different edges, not a second code path.
    concat_bands5z 5 x 90   the same five bands, each z-scored to equal weight. The
                            single biggest lever in the feature definition: raw against
                            z-scored changes the partition more (ARI 0.37) than changing
                            the algorithm does.
    concat_raw    129 x 900 full-resolution ERSP, stitched (baseline; very high-dim)

Feed the flattened matrix to lf_cluster_run.fit_and_save(feature_set='concat_*').
The orchestrator is representation-agnostic, so everything downstream (index.json
registry, MOBA, 211 validation, 252 recon) works unchanged.

NB for 213 ranking: the condition-selectivity axis is meaningless here (every sample
spans all three conditions) and degrades to n_conditions=0 -- drop or replace that
axis when ranking a concat_* run.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from functions.lf_dataset import (prepare_dataset, is_non_neural_electrode,
                                  is_micro_electrode, is_grid_electrode)
from functions.lf_features import (FREQ_BANDS_15_TO_400HZ, FREQ_BANDS_5_TO_400HZ,
                                   downsample_ersp_to_bands)
from functions.lf_hg import build_hg_feature_matrix

DEFAULT_CONDITIONS: Tuple[str, ...] = ("audio", "picture", "reading")
# Subdural GRID contacts are excluded from the concatenated track: a grid contact is
# physically a different measurement from a depth contact (larger surface area, cortical
# surface potential rather than intraparenchymal), so its ERSP is not comparable.
#
# The exclusion is at CONTACT level wherever possible, because the sample unit is one
# ELECTRODE. A depth contact in a mixed-implant patient is exactly as comparable to
# other depth contacts as anyone else's, so throwing the whole patient away costs real
# data for no methodological gain.
#
#   EL044     ECoG THROUGHOUT (Pa 51, T 46, P 6, postP 5) — no depth contacts to keep,
#             so it stays a whole-patient exclusion.
#   PAT_3415  MIXED IMPLANT, excluded as a WHOLE PATIENT from 2026-09-06 (Lora).
#             It is the only patient carrying both a subdural grid and depth shafts:
#             64 grid contacts (GA..GH) and 57 depth contacts, of which IMG, TA and
#             IPG are blacklisted as noisy, leaving OI, OS, TM, TP - 18 electrodes
#             through the gate. The contact-level split above kept those 18; the
#             decision now is that a mixed implant does not enter the cohort at all.
#             GRID_SHAFTS and NOISY_SHAFTS still list it, and are simply not reached
#             while the patient is excluded - so removing it from this tuple restores
#             the previous behaviour exactly.
#
# Pass exclude_patients=() to keep the ECoG and mixed-implant patients as well.
DEFAULT_EXCLUDE_PATIENTS: Tuple[str, ...] = ("EL044", "PAT_3415")
DEFAULT_FMAX = 500.0
DEFAULT_HG_BAND = (70.0, 150.0)
DEFAULT_DS_TIME_BINS = 30
# Ungated source cache (separate from the gated canonical cache: different params).
#
# VERSIONED, because params.json does not record WHICH patients went into a cube:
# it captures the build parameters, so a cache stays "valid" even after the set of
# available patients changes. The 08-03 cache (concat_source) was built while the
# dash rule in lf_dataset was silently discarding EL034 entirely and 4 of EL046's
# contacts, so it holds 26 patients / 8361 rows and would have been re-used
# unchanged after that fix. Bump the suffix whenever the eligible cohort changes;
# old caches are kept so any published run can still be reproduced.
#   concat_source     2026-08-03  26 patients, 8361 rows  (pre dash-fix)
#   concat_source_v2  2026-08-17  + EL034, + EL046's Fp_L-5..8   (deleted 2026-08-28)
#   concat_source_v3  2026-08-25  the split-half bug - 19,380 phantom rows (deleted)
#   concat_source_v4  2026-08-26  27 patients, 1693 gated / 2959 ungated
#   concat_source_v5  2026-09-06  + EL048 (its ERSP cubes were written 09-05),
#                     - PAT_3415 (excluded above; still IN the cache, dropped by
#                     build_concat_dataset). Built by rebuild_concat_cache.py.
# v2 and v3 were removed on 2026-08-28: v3 was the cohort with split-half files
# counted as electrodes, and keeping a default pointed at a deleted directory would
# fail at the first caller that did not pass cache_dir explicitly.
#
# THE DEFAULT IS RESOLVED, NOT PINNED. It was the literal "concat_source_v4", so the
# moment rebuild_concat_cache.py wrote a new version every caller went on reading the
# old cohort - silently, because a cache that exists is always a valid cache. The
# newest concat_source_v<N> on disk is by construction the one that script last
# built; make_cluster_statistics.py already derives its provenance string the same
# way. Sorted NUMERICALLY, so v10 beats v9. Set LF_CONCAT_CACHE to an absolute path
# to pin an older cohort when reproducing a published run.
def _newest_concat_cache() -> Path:
    env = os.environ.get("LF_CONCAT_CACHE")
    if env:
        return Path(env)
    d = Path(__file__).resolve().parents[1] / "outputs" / "_dataset"
    found = []
    if d.is_dir():
        for c in d.glob("concat_source_v*"):
            if c.is_dir() and c.name[len("concat_source_v"):].isdigit():
                found.append((int(c.name[len("concat_source_v"):]), c))
    return max(found)[1] if found else d / "concat_source_v4"


DEFAULT_CONCAT_CACHE = _newest_concat_cache()


def normalize_label(s) -> str:
    """'aH_R-1' -> 'AHR1'. Same rule the coords/recon side uses, so joins line up."""
    if s is None:
        return ""
    return str(s).replace("_", "").replace("-", "").upper()


def build_concat_dataset(
    input_dir,
    *,
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
    require_high_activity: bool = True,
    exclude_patients: Sequence[str] = DEFAULT_EXCLUDE_PATIENTS,
    cache_dir: Optional[Path] = DEFAULT_CONCAT_CACHE,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """One concatenated ERSP per electrode.

    Returns
    -------
    df_contacts : one row per kept electrode. Carries patient_id / electrode /
        contact_norm, the per-condition file paths, n_high_activity, and
        condition='concat' (so MOBA's condition view renders one category
        instead of looking broken).
    X_concat : (n_contacts, n_freq, len(conditions) * n_time) float32,
        blocks in `conditions` order.
    """
    conditions = list(conditions)
    # Ungated on purpose: the gate is applied per-ELECTRODE below, so a contact that
    # responds in only one condition still contributes its full profile.
    df_meta, _ersp_list, X_3d = prepare_dataset(
        input_dir, apply_high_activity=False, cache_dir=cache_dir, verbose=verbose
    )
    df = df_meta.reset_index(drop=True).copy()
    df["_row"] = np.arange(len(df))
    df["contact_norm"] = df["electrode"].map(normalize_label)

    # Re-apply the non-neural filter here as well. prepare_dataset already does it, but a
    # dataset CACHE written before the filter was widened would still contain e.g. AINP
    # analog-input channels — this makes a stale cache harmless.
    bad = df["electrode"].map(is_non_neural_electrode)
    if bad.any():
        if verbose:
            print(f"[lf_concat] dropped {int(bad.sum())} non-neural rows "
                  f"({sorted(df.loc[bad, 'electrode'].astype(str).unique())[:6]})")
        df = df[~bad].reset_index(drop=True)

    grid = df.apply(lambda r: is_grid_electrode(r["electrode"], r["patient_id"]), axis=1)
    if grid.any():
        if verbose:
            by = df.loc[grid].groupby("patient_id")["electrode"].size().to_dict()
            print(f"[lf_concat] dropped {int(grid.sum())} subdural GRID contacts {by} "
                  f"— their depth contacts are kept")
        df = df[~grid].reset_index(drop=True)

    mic = df.apply(lambda r: is_micro_electrode(r["electrode"], r["patient_id"]), axis=1)
    if mic.any():
        if verbose:
            sh = sorted({str(e).replace("_", "").upper().rstrip("0123456789")
                         for e in df.loc[mic, "electrode"]})
            print(f"[lf_concat] dropped {int(mic.sum())} microelectrode rows ({', '.join(sh)})")
        df = df[~mic].reset_index(drop=True)

    excl = {str(p) for p in (exclude_patients or ())}
    if excl:
        drop = df["patient_id"].astype(str).isin(excl)
        if verbose:
            present = sorted(set(df.loc[drop, "patient_id"].astype(str)))
            print(f"[lf_concat] excluded patients {sorted(excl)}"
                  f" — {int(drop.sum())} rows removed"
                  f"{'' if present else ' (none present in this dataset)'}")
        df = df[~drop].reset_index(drop=True)

    n_freq, n_time = X_3d.shape[1], X_3d.shape[2]
    keep_rows: List[List[int]] = []
    records: List[dict] = []
    n_missing_cond = n_gated_out = 0

    for (pid, con), grp in df.groupby(["patient_id", "contact_norm"], sort=True):
        by_cond = {}
        for cond in conditions:
            sub = grp[grp["condition"] == cond]
            if len(sub):
                by_cond[cond] = sub.iloc[0]
        if len(by_cond) != len(conditions):        # needs ALL conditions
            n_missing_cond += 1
            continue
        n_high = int(sum(bool(by_cond[c].get("high_activity", False)) for c in conditions))
        if require_high_activity and n_high == 0:  # responsive in >= 1 condition
            n_gated_out += 1
            continue
        keep_rows.append([int(by_cond[c]["_row"]) for c in conditions])
        rec = {
            "patient_id": pid,
            "contact_norm": con,
            "electrode": str(by_cond[conditions[0]]["electrode"]),
            "task": by_cond[conditions[0]].get("task", ""),
            "condition": "concat",                 # keeps MOBA's condition view sane
            "n_high_activity": n_high,
            # first block's path doubles as the generic `file_path` some viewers expect
            "file_path": by_cond[conditions[0]].get("file_path", ""),
        }
        for c in conditions:
            rec[f"file_path_{c}"] = by_cond[c].get("file_path", "")
        records.append(rec)

    if not records:
        raise ValueError("no electrode passed the concat filter")

    df_contacts = pd.DataFrame.from_records(records).reset_index(drop=True)
    df_contacts.insert(0, "sample_idx", np.arange(len(df_contacts)))
    X_concat = np.empty((len(keep_rows), n_freq, n_time * len(conditions)), dtype=np.float32)
    for i, rows in enumerate(keep_rows):
        X_concat[i] = np.concatenate([X_3d[r] for r in rows], axis=1)

    if verbose:
        print(f"[lf_concat] {len(df_contacts)} electrodes "
              f"({'+'.join(conditions)}) · X_concat={X_concat.shape}")
        print(f"[lf_concat]   dropped {n_missing_cond} (missing a condition), "
              f"{n_gated_out} (no high-activity condition)")
        print(f"[lf_concat]   high-activity in 1/2/3 conditions: "
              + " / ".join(str(int((df_contacts['n_high_activity'] == k).sum())) for k in (1, 2, 3)))
    return df_contacts, X_concat


# ============================================================
# Feature matrices (flattened, ready for fit_and_save)
# ============================================================
def concat_hg_features(X_concat: np.ndarray, *, hg_band=DEFAULT_HG_BAND,
                       fmax: float = DEFAULT_FMAX) -> np.ndarray:
    """(n, 1 x 3*n_time): HG-band mean per time bin. Concatenation is along time, so
    the frequency axis is untouched and the standard HG extractor applies directly."""
    return build_hg_feature_matrix(list(X_concat), hg_band=hg_band, fmax=fmax)


def concat_rawds_features(X_concat: np.ndarray, *, n_blocks: int = 3,
                          freq_band_edges: Iterable = FREQ_BANDS_15_TO_400HZ,
                          fmax_hz: float = DEFAULT_FMAX,
                          time_bins_out: int = DEFAULT_DS_TIME_BINS) -> np.ndarray:
    """(n, 15 * n_blocks*30): each condition block downsampled to 15 bands x 30 bins,
    then re-stitched. Downsamples PER BLOCK (not across the seam) using the same
    routine stage-04 pooling uses, so the grid is identical to the pooling DS grid."""
    n, _, n_time_total = X_concat.shape
    nb = n_time_total // n_blocks
    out = []
    for i in range(n):
        blocks = [
            downsample_ersp_to_bands(X_concat[i][:, b * nb:(b + 1) * nb], freq_band_edges,
                                     fmax_hz=fmax_hz, time_bins_out=time_bins_out)
            for b in range(n_blocks)
        ]
        out.append(np.concatenate(blocks, axis=1))
    return np.stack(out).reshape(n, -1).astype(np.float32)


def concat_bands5_features(X_concat: np.ndarray, *, n_blocks: int = 3,
                           fmax_hz: float = DEFAULT_FMAX,
                           time_bins_out: int = DEFAULT_DS_TIME_BINS) -> np.ndarray:
    """(n, 5 * n_blocks*30): concat_rawds coarsened from 15 bands to 5.

    SAME BUILDER, DIFFERENT EDGES. This is concat_rawds_features with
    FREQ_BANDS_5_TO_400HZ, which matters for two reasons. It averages the ORIGINAL
    frequency bins rather than averaging the 15 pre-averaged bands, so a wide band is
    weighted by the number of bins it actually contains instead of giving a 3 Hz
    sub-band the same say as a 40 Hz one. And every 5-band edge lands on a 15-band
    edge, so the two feature sets are nested and any difference between them is the
    resolution and nothing else.

    Why five and why these edges is argued at FREQ_BANDS_5_TO_400HZ; the short version
    is that it is indistinguishable from all 15 bands on anatomical coherence at a
    third of the features, while four bands is measurably worse.
    """
    return concat_rawds_features(X_concat, n_blocks=n_blocks,
                                 freq_band_edges=FREQ_BANDS_5_TO_400HZ,
                                 fmax_hz=fmax_hz, time_bins_out=time_bins_out)


def concat_bands5z_features(X_concat: np.ndarray, *, n_blocks: int = 3,
                            fmax_hz: float = DEFAULT_FMAX,
                            time_bins_out: int = DEFAULT_DS_TIME_BINS) -> np.ndarray:
    """concat_bands5 with each band z-scored to equal weight across the cohort.

    WHAT THIS FIXES. Euclidean distance has no idea that 1/f exists. In concat_rawds the
    four lowest of fifteen bands hold 55% of the total sum of squares, and in
    concat_bands5 the 1-20 Hz band alone holds 44%, so k-means spends most of its budget
    on low-frequency power whether or not that is where the structure is. Measured: the
    partition follows 8-13 / 13-20 / 4-8 Hz when the features are raw, and 270-320 /
    170-220 / 220-270 Hz once the bands are equalised.

    HOW BIG THE EFFECT IS. Raw against z-scored gives ARI 0.37 on the same electrodes at
    the same K - a LARGER change than swapping the algorithm (k-means against Ward and
    convex NMF agree at 0.25-0.36). Normalisation is not a detail here; it is the biggest
    single lever in the feature definition, which is why it gets its own feature set
    rather than a flag on another one.

    ONE MEAN AND ONE SD PER BAND, over every electrode, condition and time bin - a
    COHORT-LEVEL transform, so it is deterministic given the cohort and identical for
    every electrode. Per-electrode z-scoring would be a different thing entirely: it
    would erase how strongly a contact responds, which is a large part of what
    distinguishes the clusters.

    THE UNITS ARE NO LONGER dB. Values are standard deviations within a band, so a
    centroid heatmap of this feature set must not be read against a dB scale bar.
    """
    X = concat_bands5_features(X_concat, n_blocks=n_blocks, fmax_hz=fmax_hz,
                               time_bins_out=time_bins_out).astype(np.float64)
    n_cols = n_blocks * time_bins_out
    n_bands = X.shape[1] // n_cols
    for b in range(n_bands):
        sl = slice(b * n_cols, (b + 1) * n_cols)
        blk = X[:, sl]
        X[:, sl] = (blk - blk.mean()) / max(blk.std(), 1e-12)
    return X.astype(np.float32)


def concat_raw_features(X_concat: np.ndarray) -> np.ndarray:
    """(n, n_freq * 3*n_time): full-resolution flatten. Very high-dimensional —
    the per-condition `raw` track already scores worst on silhouette; kept as the
    curse-of-dimensionality baseline."""
    return X_concat.reshape(X_concat.shape[0], -1).astype(np.float32)


def concat_feature_names(kind: str, *, n_blocks: int = 3,
                         conditions: Sequence[str] = DEFAULT_CONDITIONS,
                         n_time_block: int = 300) -> List[str]:
    """Human-readable column names, e.g. 'audio|70-150Hz|t012'."""
    conds = list(conditions)[:n_blocks]
    if kind in ("concat_hg", "concat_hg_all"):
        # concat_hg_all is the SAME representation on the ungated electrode set, so the
        # columns are identical and every downstream reshape keeps working.
        return [f"{c}|hg|t{t:03d}" for c in conds for t in range(n_time_block)]
    if kind in ("concat_rawds", "concat_bands5", "concat_bands5z"):
        # bands5z is bands5 rescaled, so the columns are identical and every reshape,
        # grid and centroid renderer downstream keeps working unchanged
        edges = (FREQ_BANDS_5_TO_400HZ if kind.startswith("concat_bands5")
                 else FREQ_BANDS_15_TO_400HZ)
        bands = [f"{int(lo)}-{int(hi)}" for lo, hi in edges]
        return [f"{c}|{b}Hz|t{t:02d}" for b in bands for c in conds
                for t in range(DEFAULT_DS_TIME_BINS)]
    if kind == "concat_raw":
        return [f"{c}|f{f:03d}|t{t:03d}" for f in range(129) for c in conds
                for t in range(n_time_block)]
    raise ValueError(f"unknown kind {kind!r}")
