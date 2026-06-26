"""
lf_trials_qc.py
===============

Readable reimplementation of ``extract_trials_with_qc`` — the canonical trial-timing
+ per-condition QC step used by the MicroEPI notebooks (11 / 13). This replaces the
compiled ``lf_micromacro_full.pyc``.

Why this is safe to rewrite
---------------------------
The *hard part* — photodiode onset/offset detection, trial pairing, fake/invalid
handling, trial-end (``duration_end_abs``) computation, and the QC plot — lives in
``LFfunctions_PDextract.get_trigger_indexes_photodiode``, which is intact, readable
source. Only the thin **wrapper** around it (wrong-response filter + per-condition
hard/IQR filtering + grouping + QC TSV) was lost with the deleted ``lf_micromacro``.
This module reimplements exactly that wrapper and calls the same driver with the
same arguments, so the heavy lifting is byte-for-byte the original behaviour.

Output contract (what the notebook consumes)
--------------------------------------------
extract_trials_with_qc(...) -> dict with:
    cond_groups : {cond_name: (ons, offs, tends)}  arrays in SAMPLE units of fs
    on_abs, off_abs : (n_trials,) int64 sample indices
    metrics : the driver's metrics dict (incl. duration_end_abs)
    trial_ids_all, invalid_merged, invalid_from_tsv, fs

Notes
-----
* The driver is called with only the kwargs its current signature accepts (it is
  introspected), so signature drift (e.g. the old ``baseline_window`` args) can't
  break the call.
* Hard QC: keep trials with ``stim_s >= min_stim_s`` and ``0 <= post_s <= max_post_s``.
  IQR QC: within each condition, keep ``post_s`` inside ``[Q1-k*IQR, Q3+k*IQR]``.
  These match the recovered logic; validate trial counts on your first run.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

__all__ = ["extract_trials_with_qc", "DEFAULT_WRONG_VALUES"]

# Response-column values that mark an incorrect/invalid behavioural trial.
# Override per call via `wrong_values=` / `extra_wrong=` if your TSV uses other tokens.
DEFAULT_WRONG_VALUES = frozenset({
    "incorrect", "wrong", "false", "0", "no", "miss", "missed",
    "timeout", "no_response", "none", "nan", "",
})


# -----------------------------------------------------------------------------
# small helpers (faithful to the recovered originals)
# -----------------------------------------------------------------------------
def _find_tsv_header_row(path, needles, max_scan=40):
    """Return the 0-based line index of the header row (the first row, within the
    first ``max_scan`` lines, that contains any of ``needles`` as a tab cell), or None."""
    needles_l = {str(n).lower() for n in needles}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for k, line in enumerate(fh):
            if k > max_scan:
                break
            cells = [c.strip().lower() for c in line.rstrip("\n").split("\t")]
            if any(nl in cells for nl in needles_l):
                return k
    return None


def _iqr_keep(post_s, k):
    """Tukey-fence keep-mask on ``post_s``: keep values in [Q1-k*IQR, Q3+k*IQR].
    Returns (keep_mask, (lo, hi)). With <2 samples, keeps all (fence = nan)."""
    post_s = np.asarray(post_s, dtype=float)
    if post_s.size < 2:
        return np.ones_like(post_s, dtype=bool), (float("nan"), float("nan"))
    q1, q3 = np.percentile(post_s, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return ((post_s >= lo) & (post_s <= hi)), (float(lo), float(hi))


def _alias_cond(label, cond_alias=None):
    """Normalise a condition label; apply cond_alias mapping if provided."""
    s = str(label).strip().lower()
    if isinstance(cond_alias, dict):
        return cond_alias.get(s, cond_alias.get(label, s))
    return s


def _unique_in_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def extract_trials_with_qc(d_all, preset, tsv_path, *, pid,
                           baseline_w=(-0.8, 0.0), min_stim_s=0.5, max_post_s=10.0,
                           iqr_k=1.5, pd_threshold=0.4, response_col="response_type",
                           wrong_values=None, extra_wrong=None, cond_alias=None,
                           save_qc_tsv=True, qc_dir=None):
    """PD extraction + wrong-response filter + per-condition hard/IQR filtering.

    See module docstring for the output contract.
    """
    import inspect
    from LFfunctions_PDextract import get_trigger_indexes_photodiode

    fs = float(d_all["fs"])
    trial_ids_all = list(preset.get("trial_ids", []) or [])
    fake_list_raw = list(preset.get("fake_trials", []) or [])
    invalid_preset = list(preset.get("invalid_trials", []) or [])

    wrong_set = set(wrong_values) if wrong_values is not None else set(DEFAULT_WRONG_VALUES)
    if extra_wrong:
        wrong_set |= set(extra_wrong)

    # ---- wrong-response trials from the behavioural TSV ----
    invalid_from_tsv = []
    if tsv_path and os.path.exists(tsv_path):
        hdr = _find_tsv_header_row(tsv_path, needles=[response_col, "category", "blockname"])
        if hdr is None:
            print(f"  [{pid}] WARNING: could not auto-find header row with {response_col!r}")
        else:
            df = pd.read_csv(tsv_path, sep="\t", header=hdr)
            if response_col not in df.columns:
                print(f"  [{pid}] WARNING: response col {response_col!r} not in TSV columns "
                      f"{list(df.columns)}")
            else:
                vals = df[response_col].astype(str).str.lower().str.strip()
                is_wrong = vals.isin(wrong_set)
                invalid_from_tsv = [int(i) for i in np.where(is_wrong.to_numpy())[0]]
                print(f"  [{pid}] wrong-tagged: {len(invalid_from_tsv)}")
    else:
        print(f"  [{pid}] no TSV at {tsv_path} -> no wrong-response filter")

    invalid_merged = sorted(set(invalid_preset) | set(invalid_from_tsv))
    print(f"  [{pid}] invalid merged: preset={len(invalid_preset)} + tsv={len(invalid_from_tsv)} "
          f"-> total={len(invalid_merged)}")

    # ---- photodiode detection via the (intact) driver ----
    raw_pd = np.asarray(d_all["photodiode"], dtype=np.float32).reshape(-1, 1)
    trig_name = preset.get("trig", "photodiode")
    pd_names = [trig_name]

    driver_kwargs = dict(
        raw_signals=raw_pd, sampling_rate=fs, channel_names=pd_names,
        trig_name=trig_name, time_range=preset.get("time_range", (0, -1)),
        threshold_val=pd_threshold, flip_trigs=preset.get("flip", False),
        do_plot=True,
        plot_title=f"{pid} - photodiode + TSV pairing (fs={fs:.0f} Hz)",
        plot_kwargs={"color": "black"},
        trial_ids=trial_ids_all, invalid_trials=invalid_merged, ignore_invalid=True,
        fake_trials=fake_list_raw, extra_table_path=tsv_path,
        manual_trigs_path=preset.get("manual_trig"),
        return_extra_metrics=True,
    )
    # pass only kwargs the current driver accepts (robust to signature drift)
    accepted = set(inspect.signature(get_trigger_indexes_photodiode).parameters)
    driver_kwargs = {k: v for k, v in driver_kwargs.items() if k in accepted}

    out = get_trigger_indexes_photodiode(**driver_kwargs)
    if isinstance(out, tuple) and len(out) == 5:
        on_abs, off_abs, _valid_on, _valid_off, metrics = out
    elif isinstance(out, tuple) and len(out) == 3:
        on_abs, off_abs, metrics = out
    else:
        on_abs, off_abs = out[0], out[1]
        metrics = {}

    on_abs = np.asarray(on_abs, dtype=np.int64)
    off_abs = np.asarray(off_abs, dtype=np.int64)

    # ---- trial ends (duration_end_abs) ----
    raw_tends = metrics.get("duration_end_abs") if isinstance(metrics, dict) else None
    if raw_tends is None:
        raise RuntimeError(
            f"[{pid}] behavioral TSV did not produce duration_end_abs. Check that "
            f"{os.path.basename(tsv_path) if tsv_path else '<no tsv>'} has a usable "
            f"'duration' or 'response_time' column.")
    tends_abs = np.asarray(raw_tends, dtype=np.int64)
    _none_idx = [i for i, v in enumerate(raw_tends) if v is None]
    if _none_idx:
        raise RuntimeError(f"[{pid}] behavioral TSV is missing a trial-end for "
                           f"{len(_none_idx)} trial(s). first missing: {_none_idx[:10]}")

    # ---- align trial_ids to the detected trials (driver already removed fakes) ----
    n = min(len(on_abs), len(off_abs), len(tends_abs))
    if trial_ids_all and len(trial_ids_all) != n:
        print(f"  [{pid}] note: trial_ids ({len(trial_ids_all)}) != detected trials ({n}); "
              f"aligning to the first {n}")
    ids = [_alias_cond(t, cond_alias) for t in trial_ids_all[:n]] if trial_ids_all \
        else [f"cond{0}"] * n
    on_abs, off_abs, tends_abs = on_abs[:n], off_abs[:n], tends_abs[:n]
    invalid_set = set(invalid_merged)

    # ---- per-condition hard + IQR QC ----
    print(f"  [{pid}] min_stim={min_stim_s}s max_post={max_post_s}s iqr_k={iqr_k}")
    cond_groups, qc_rows = {}, []
    for cond in _unique_in_order(ids):
        idx = np.array([i for i in range(n) if ids[i] == cond], dtype=int)
        if idx.size == 0:
            continue
        ons, offs, tends = on_abs[idx], off_abs[idx], tends_abs[idx]
        stim_s = (offs - ons) / fs
        post_s = (tends - offs) / fs

        not_invalid = np.array([i not in invalid_set for i in idx], dtype=bool)
        keep_hard = (stim_s >= min_stim_s) & (post_s >= 0.0) & (post_s <= max_post_s) & not_invalid
        n_hard = int(keep_hard.sum())

        keep = keep_hard.copy()
        iqr_lo = iqr_hi = float("nan")
        if n_hard >= 2:
            keep_iqr_sub, (iqr_lo, iqr_hi) = _iqr_keep(post_s[keep_hard], iqr_k)
            keep[np.flatnonzero(keep_hard)] &= keep_iqr_sub
        n_iqr = int(keep.sum())

        cond_groups[cond] = (ons[keep].astype(np.int64),
                             offs[keep].astype(np.int64),
                             tends[keep].astype(np.int64))

        for j, i in enumerate(idx):
            qc_rows.append(dict(cond=cond, idx=int(i),
                                stim_s=float(stim_s[j]), post_s=float(post_s[j]),
                                kept=bool(keep[j]), hard_ok=bool(keep_hard[j])))

        print(f"    {cond}: kept {n_iqr:3d}/{idx.size}  hard={n_hard}  iqr={n_iqr}  "
              f"fence=[{iqr_lo:.2f},{iqr_hi:.2f}]s")

    # ---- QC TSV ----
    if save_qc_tsv and qc_rows:
        out_dir = qc_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        qc_path = os.path.join(out_dir, f"{pid}_trial_QC_IQR.tsv")
        pd.DataFrame(qc_rows).to_csv(qc_path, sep="\t", index=False)
        print(f"  [{pid}] QC TSV -> {qc_path}")

    return dict(cond_groups=cond_groups, on_abs=on_abs, off_abs=off_abs,
                metrics=metrics, trial_ids_all=trial_ids_all,
                invalid_merged=invalid_merged, invalid_from_tsv=invalid_from_tsv, fs=fs)
