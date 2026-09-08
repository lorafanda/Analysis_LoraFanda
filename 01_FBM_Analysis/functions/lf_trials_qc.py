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
# ---------------------------------------------------------------------------
# Per-block photodiode equalisation
# ---------------------------------------------------------------------------
def _pd_first_pass(pd_1d, fs, preset, threshold):
    """Onsets used ONLY to place block edges, found deliberately permissively.

    At the working threshold this pass is circular: on a fading photodiode it misses the
    dim block, so that block gets no window and is never equalised - which is the whole
    point of the exercise. It does not have to be right about trials, only about where
    the blocks are, so it runs at a quarter of the working threshold (floor 0.05).
    Measured on the synthetic drift: 0.4 finds 20 of 30 onsets, 0.2 finds 23, 0.1 finds
    all 30 with nothing spurious; below 0.05 it starts inventing edges.
    """
    from LFfunctions_PDextract import _pd_detect_deriv_core
    t0, t1 = preset.get("time_range", (0, -1))
    i0 = max(0, int(t0 * fs))
    i1 = -1 if (t1 is None or t1 < 0) else min(len(pd_1d), int(t1 * fs))
    loose = max(0.05, float(threshold) / 4.0)
    on, _off = _pd_detect_deriv_core(
        pd_1d, t_start=i0, t_end=(None if i1 < 0 else i1), sampling_rate=fs,
        threshold_val=loose, flip_trigs=False, do_plot=False)
    return list(map(int, on))


def _block_windows_from_runs(on_abs, trial_ids, n_samples, fs, pad_s=2.0):
    """Sample windows for each contiguous run of trial_ids, from a first detection pass.

    trial_ids is the condition per trial in order, so its runs ARE the blocks. The
    onsets are taken in the same order; a run therefore spans from its first onset to
    its last, padded. Returns [] when the two cannot be lined up, and the caller then
    leaves the trace alone rather than guessing.
    """
    import itertools
    if len(on_abs) == 0 or not trial_ids:
        return []
    runs = [(k, len(list(g))) for k, g in itertools.groupby(trial_ids)]
    if len(runs) < 2:
        return []
    pad = int(pad_s * fs)
    out, i = [], 0
    for _name, ln in runs:
        j = min(i + ln, len(on_abs)) - 1
        if j < i:
            break
        t0 = max(0, int(on_abs[i]) - pad)
        t1 = min(n_samples, int(on_abs[j]) + pad)
        out.append((t0, t1))
        i += ln
    # blocks must be ordered and non-degenerate to be worth using
    return out if all(b > a for a, b in out) else []


def _equalise_blocks(x, windows, *, label="", floor=0.15):
    """Centre and scale each block so the steps are the same height everywhere.

    The scale is the block's 5-95 percentile spread, which is the height of a photodiode
    step and is not moved by a single spike the way peak-to-peak is. A block quieter
    than `floor` of the loudest is NOT amplified to match - that would turn its noise
    into edges - it is left at the floor and reported, because a block that dim is a
    finding rather than something to normalise away.
    """
    import numpy as np
    y = np.array(x, dtype=np.float32, copy=True)
    spreads = []
    for (a, b) in windows:
        seg = x[a:b]
        lo, hi = np.percentile(seg, [5, 95])
        spreads.append(float(hi - lo))
    ref = max(spreads) if spreads else 0.0
    if ref <= 0:
        return y, []
    report = []
    for (a, b), sp in zip(windows, spreads):
        seg = x[a:b]
        med = float(np.median(seg))
        rel = sp / ref
        use = max(sp, floor * ref)
        y[a:b] = (seg - med) / (use + 1e-12)
        report.append((a, b, sp, rel, rel < floor))
    if label:
        print(f"  [{label}] photodiode equalised over {len(windows)} blocks "
              f"(5-95 spread, relative to the loudest):")
        for k, (a, b, sp, rel, clipped) in enumerate(report):
            print(f"      block {k}: {a}-{b} samples   spread {sp:.4g}   "
                  f"{rel:.2f} x the loudest" + ("   FLOORED, this block is very dim"
                                                if clipped else ""))
    return y, report


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
    # THE FLIP IS APPLIED HERE, NOT PASSED DOWN, AND THAT IS DELIBERATE. The driver
    # flips inside _pd_detect_deriv_core (x = -x) but plots pd_full, the array it was
    # given - so passing flip_trigs moved the onsets and left the figure identical,
    # which reads as the setting being ignored. Negating first is the same arithmetic
    # (the driver flips straight after slicing, before any filtering) and the trace
    # that is plotted is then the trace that was detected on.
    flip_pd = bool(preset.get("flip", False))
    pd_1d = np.asarray(d_all["photodiode"], dtype=np.float32)
    if flip_pd:
        pd_1d = -pd_1d
    print(f"  [{pid}] photodiode flip = {flip_pd}  (from the preset; the figure shows "
          f"the flipped trace)")
    # ---- per-block equalisation, when the preset asks for it ----
    # The threshold is a fraction of the sharpest edge in the window, so a file whose
    # photodiode drifts cannot be served by one number. Equalising each block first
    # makes that one number mean the same thing from start to end.
    pd_blocks = preset.get("pd_blocks")
    if pd_blocks:
        n_samp = len(pd_1d)
        if pd_blocks == "auto":
            first = _pd_first_pass(pd_1d, fs, preset, pd_threshold)
            wins = _block_windows_from_runs(first, trial_ids_all, n_samp, fs)
            if not wins:
                # EQUAL PARTS RATHER THAN NOTHING. Cruder than real block edges, but the
                # scaling only has to be block-ish, and leaving a dim third of the file
                # on a threshold set by a bright third is the failure being fixed.
                import itertools as _it
                n_runs = len({k for k, _ in _it.groupby(trial_ids_all)}) or 3
                n_runs = max(2, len([1 for _k, _g in _it.groupby(trial_ids_all)]) or 3)
                t0, t1 = preset.get("time_range", (0, -1))
                a = max(0, int(t0 * fs))
                b = n_samp if (t1 is None or t1 < 0) else min(n_samp, int(t1 * fs))
                edges = [int(a + (b - a) * k / n_runs) for k in range(n_runs + 1)]
                wins = list(zip(edges[:-1], edges[1:]))
                print(f"  [{pid}] pd_blocks='auto': could not place block edges from "
                      f"{len(first)} first-pass onsets and {len(trial_ids_all)} "
                      f"trial_ids - falling back to {n_runs} EQUAL parts of the window")
        else:
            wins = [(max(0, int(t0 * fs)), min(n_samp, int(t1 * fs)))
                    for t0, t1 in pd_blocks]
        if wins:
            pd_1d, _ = _equalise_blocks(pd_1d, wins, label=pid)

    raw_pd = pd_1d.reshape(-1, 1)
    trig_name = preset.get("trig", "photodiode")
    pd_names = [trig_name]

    driver_kwargs = dict(
        raw_signals=raw_pd, sampling_rate=fs, channel_names=pd_names,
        trig_name=trig_name, time_range=preset.get("time_range", (0, -1)),
        threshold_val=pd_threshold, flip_trigs=False,   # already applied above
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
