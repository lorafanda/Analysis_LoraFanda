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
    concat_raw    129 x 900 full-resolution ERSP, stitched (baseline; very high-dim)

Feed the flattened matrix to lf_cluster_run.fit_and_save(feature_set='concat_*').
The orchestrator is representation-agnostic, so everything downstream (index.json
registry, MOBA, 211 validation, 252 recon) works unchanged.

NB for 213 ranking: the condition-selectivity axis is meaningless here (every sample
spans all three conditions) and degrades to n_conditions=0 -- drop or replace that
axis when ranking a concat_* run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from functions.lf_dataset import prepare_dataset, is_non_neural_electrode, is_micro_electrode
from functions.lf_features import FREQ_BANDS_15_TO_400HZ, downsample_ersp_to_bands
from functions.lf_hg import build_hg_feature_matrix

DEFAULT_CONDITIONS: Tuple[str, ...] = ("audio", "picture", "reading")
# Grid (ECoG) patients are excluded from the concatenated track: the electrode geometry
# and coverage differ from the depth/sEEG cohort, so a per-electrode functional TYPE is
# not comparable across the two. Pass exclude_patients=() to keep them.
#
#   EL044     whole-patient ECoG — 108 contacts over 4 shafts (Pa 51, T 46, P 6, postP 5)
#   PAT_3415  carries a 64-contact grid (GA..GH, 8 shafts x 8) alongside its depth
#             electrodes. The patient is excluded ENTIRELY, not just the G* shafts:
#             mixing grid and depth contacts from one subject would put two different
#             recording geometries into the same per-electrode sample space.
DEFAULT_EXCLUDE_PATIENTS: Tuple[str, ...] = ("EL044", "PAT_3415")
DEFAULT_FMAX = 500.0
DEFAULT_HG_BAND = (70.0, 150.0)
DEFAULT_DS_TIME_BINS = 30
# Ungated source cache (separate from the gated canonical cache: different params).
DEFAULT_CONCAT_CACHE = Path(__file__).resolve().parents[1] / "outputs" / "_dataset" / "concat_source"


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
    if kind == "concat_hg":
        return [f"{c}|hg|t{t:03d}" for c in conds for t in range(n_time_block)]
    if kind == "concat_rawds":
        bands = [f"{int(lo)}-{int(hi)}" for lo, hi in FREQ_BANDS_15_TO_400HZ]
        return [f"{c}|{b}Hz|t{t:02d}" for b in bands for c in conds
                for t in range(DEFAULT_DS_TIME_BINS)]
    if kind == "concat_raw":
        return [f"{c}|f{f:03d}|t{t:03d}" for f in range(129) for c in conds
                for t in range(n_time_block)]
    raise ValueError(f"unknown kind {kind!r}")
