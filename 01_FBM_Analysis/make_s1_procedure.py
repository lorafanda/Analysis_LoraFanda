#!/usr/bin/env python3
"""
make_s1_procedure.py - the stage-01 procedure, step by step, each step cited to the code that
does it. Imported by make_s1_tab.py (procedure_html); runnable on its own to check the citations.

    python make_s1_procedure.py        resolve every citation and print the table as text

WHAT A ROW IS. One thing the pipeline does to a recording, in the order it does it, written
from a reading of the function that does it - not from the docstring and not from what the
step is meant to do - followed by the path (relative to Analysis_LoraFanda), the function and
the line. Settings are read from functions/config.py at build time and printed with their
values, so a row cannot quote a number the pipeline no longer uses.

A CITATION CANNOT ROT SILENTLY. The line of a function is found by parsing the file (ast);
the line of a block inside a function by a text anchor that must occur exactly once. A
function that was renamed or an anchor that is gone stops the build with the row named,
rather than leaving a link to the wrong line.

Adding a step: a (title, text, [cites]) tuple in rows() below; a cite is (path, "function")
or (path, None, "anchor text") for a block inside a function.
"""
from __future__ import annotations

import ast
import html
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "functions"))
import config as cfg                                                  # noqa: E402

GH = "https://github.com/lorafanda/Analysis_LoraFanda/blob/main/"
P140 = "01_FBM_Analysis/140_ersp_pipeline.py"
PIO = "01_FBM_Analysis/functions/lf_io_utils.py"
PERSP = "01_FBM_Analysis/functions/lf_ersp.py"
PTR = "01_FBM_Analysis/functions/lf_trials.py"
PMM = "01_FBM_Analysis/functions/lf_micromacro.py"
PPD = "01_FBM_Analysis/LFfunctions_PDextract.py"
PCFG = "01_FBM_Analysis/functions/config.py"


@lru_cache(maxsize=None)
def _src(rel: str) -> str:
    p = REPO / rel
    if not p.exists():
        raise SystemExit(f"make_s1_procedure: cited file does not exist: {rel}")
    return p.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _defs(rel: str) -> dict:
    out = {}
    for node in ast.walk(ast.parse(_src(rel))):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out.setdefault(node.name, node.lineno)
    return out


def line_of(rel: str, symbol: str | None, anchor: str | None = None) -> int:
    if anchor is not None:
        hits = [i for i, ln in enumerate(_src(rel).splitlines(), 1) if anchor in ln]
        if len(hits) != 1:
            raise SystemExit(f"make_s1_procedure: anchor {anchor!r} occurs {len(hits)} times in {rel} (needs exactly 1)")
        return hits[0]
    d = _defs(rel)
    if symbol not in d:
        raise SystemExit(f"make_s1_procedure: no function {symbol!r} in {rel}")
    return d[symbol]


def cite(c) -> tuple[str, str]:
    """(html, plain) for one citation tuple."""
    rel, symbol = c[0], c[1]
    anchor = c[2] if len(c) > 2 else None
    n = line_of(rel, symbol, anchor)
    what = f"{symbol}()" if symbol and anchor is None else (symbol or "block")
    h = (f'<a href="{GH}{rel}#L{n}"><code>{html.escape(rel)}</code></a> &middot; '
         f'<code>{html.escape(what)}</code> &middot; L{n}')
    return h, f"{rel} · {what} · L{n}"


def _fmt_dict(d, f=lambda k, v: f"{k} {v}"):
    return ", ".join(f(k, v) for k, v in sorted(d.items())) or "none"


def rows() -> list[tuple[str, str, list]]:
    c = cfg
    hop_ms = 1000.0 * (c.nperseg - c.noverlap) / 1000.0
    df_hz = 1000.0 / c.nfft
    n_bins = int(c.fmax // df_hz) + 1
    n_stim = int(round(c.proportions[1] / sum(c.proportions) * c.n_time_bins))
    extra = _fmt_dict(getattr(c, "notch_extra_bases", {}), lambda k, v: f"{k} {', '.join(f'{b:g}' for b in v)} Hz")
    floors = _fmt_dict(getattr(c, "notch_interp_min_hw_hz", {}),
                       lambda k, v: f"{k} " + ", ".join(f"{int(f)} Hz ±{w:g}" for f, w in v.items()))
    rej = getattr(c, "ersp_trial_reject_hg_mad", {})
    return [
        ("0 · trigger tables (only with <code>--pd</code>)",
         "Not part of an ordinary run: without <code>--pd</code> the trial tables already in <code>prep0/</code> are read as they are. "
         "With it, EL / HUG: the raw file is loaded and the preset's trigger channel (<code>EL_PRESETS</code> / <code>PAT_PRESETS</code> <code>trig</code>) is handed to "
         "<code>get_trigger_indexes_photodiode</code> with threshold 0.40, the preset's <code>flip</code>, <code>time_range</code>, <code>invalid_trials</code>, <code>fake_trials</code>, "
         "the events log found in <code>raw/</code> and, when the preset names one, a manual trigger table; <code>parse_and_save</code> writes the per-condition tables into <code>prep0/</code>. "
         "MicroEPI: <code>extract_events_from_photodiode</code> on the export's photodiode with flip always on, then <code>save_onsets_offsets_by_condition</code>. "
         "A preset whose trigger channel is not in the recording is skipped with a warning.",
         [(P140, "run_pd_extraction"), (PPD, "get_trigger_indexes_photodiode"), (PPD, "parse_and_save"),
          (PMM, "extract_events_from_photodiode")]),
        ("1 · load",
         "MicroEPI ids (<code>MICROEPI_MAT_PATIENTS</code>): the preset's <code>.mat</code> files are concatenated and macros + micros stacked into one matrix with an "
         "<code>is_micro</code> mask. Everyone else: the first <code>.TRC</code> in <code>raw/</code>, else the first <code>.edf</code>, else the first <code>.h5</code> (alphabetical); "
         f"a patient in <code>RAW_CONCAT</code> ({', '.join(sorted(getattr(c, 'RAW_CONCAT', {})))}) has its listed files joined along time, and the load fails if they differ in channels or sampling rate. "
         "Units are whatever the loader returns: µV for <code>.h5</code>, <code>.TRC</code> and <code>.mat</code>, <b>volts</b> for <code>.edf</code> (read through mne: EL030, EL033–EL036) — nothing downstream depends on the unit except the Signal figures, which measure it.",
         [(P140, "_load_signals_and_prep_for_patient"), (PIO, "load_raw_for_patient"), (PIO, "load_first_raw_in_dir"),
          (PMM, "load_and_concatenate_mats"), (PMM, "build_combined_signals")]),
        ("2 · auxiliary channels out",
         "MicroEPI: every name <code>_is_non_neural</code> matches is dropped. EL / HUG: first names starting with MRK, MKR, X, ECG, EX or AUDIO, then (since 2026-10-02) every "
         "<code>_is_non_neural</code> name — EKG, EMG, EOG, TRIG, AINP, E1–E8, X-families, anything containing “+” — <b>except the photodiode</b> (a name starting with PHOTO), "
         "which stays as a figures-only channel in the four HUG recordings that have one.",
         [(P140, None, "# ── Aux drop (ECG / DC / markers)"), (PIO, "filter_aux_channels"), (PIO, "_is_non_neural"), (P140, "_is_photodiode")]),
        ("3 · EL only: which names are electrodes, and the task window",
         "For ids starting with EL, a channel is kept only if its name contains “_” or “-” (this is what removes DC6, TRIG and the scalp EEG names); a patient in "
         f"<code>EL_GRID_PATIENTS</code> keeps names starting with its <code>EL_GRID_KEEP_PREFIXES</code> and containing a digit instead. Then the signal is cut to the preset's "
         f"<code>time_range</code> when its end is &gt; 0 (trial indices are shifted by the same offset later). <code>STRIP_HEMI_PATIENTS</code> ({', '.join(sorted(c.STRIP_HEMI_PATIENTS))}): "
         "the <code>_L</code> / <code>_R</code> before the contact number is removed from every name.",
         [(P140, None, "# ── EL prefix / grid filter — EL ONLY"), (P140, None, "# ── EXPERIMENT-WINDOW CROP (EL ONLY)"),
          (P140, None, "# ── PER-PATIENT CHANNEL-NAME OVERRIDE (EL ONLY)")]),
        ("4 · contacts without anatomy out",
         "Every cohort: a contact whose <code>tissueLabel</code> in the patient's electrodes TSV is empty, NaN or starts with “unknown” (any case) is dropped; names are matched through "
         "<code>CHANNEL_SHAFT_ALIAS</code>. If the TSV is not found nothing is dropped. For a patient in <code>MIXED_GRID_DEPTH_PATIENTS</code> "
         f"({', '.join(sorted(c.MIXED_GRID_DEPTH_PATIENTS))}) the names starting with its keep-prefixes are exempt.",
         [(P140, None, '# ── DROP "UNKNOWN" PARCELLATION CHANNELS'), (PIO, "unknown_indices_for_patient"),
          (PIO, "derive_unknown_channels_from_electrodes_tsv"), (PIO, "alias_label")]),
        ("5 · bad-listed contacts out",
         "Names in <code>bad_channels_manual[patient]</code> (compared after <code>normalize_label</code>: separators removed, upper case) are removed from the signal matrix here — "
         "before the reference, the notch and every figure.",
         [(P140, None, "# ── DROP BAD-LISTED CONTACTS"), (PIO, "normalize_label"), (PCFG, None, "bad_channels_manual = {")]),
        ("6 · reference",
         "One subtraction over the whole recording. <b>EL / HUG</b>: WM contacts = <code>tissueWeights_1 &gt; 0.97</code> and first token of <code>tissueLabel</code> starting with <code>wm-</code> "
         "(or the patient's <code>MANUAL_WM_CHANNELS</code> list, or the Lookup workbook for a patient in <code>LOOKUP_ANATOMY_PATIENTS</code>), minus <code>WM_NOT_REFERENCE</code>; "
         "their <b>mean</b> is subtracted from every channel, and the patient's run ends with <code>error-reref</code> if fewer than 3 are in the recording. The reference contacts get no ERSP, figure or cube. "
         f"A patient in <code>WHOLE_CAR_PATIENTS</code> ({', '.join(sorted(c.WHOLE_CAR_PATIENTS)) or 'nobody'}) gets the mean of all neural channels instead; a grid patient in "
         "<code>GRID_CAR_PATIENTS</code> a mean per grid (the mean of each name-prefix group's good channels). <b>MicroEPI</b>: the same WM mean is subtracted from the macro contacts only; "
         "the WM contacts stay in the data; micros are left raw unless the preset names a <code>micro_reref_anchor</code>. Unlike EL / HUG, a MicroEPI patient with fewer than 3 WM "
         "contacts is <b>not</b> stopped: the macros stay unreferenced and the log line reads “0 contacts used”.",
         [(P140, None, "# ── REREFERENCING"), (P140, "apply_wm_reref"), (PERSP, "apply_wm_reference_with_exclusions"),
          (PIO, "wm_labels_for_patient"), (PIO, "derive_wm_channels_from_electrodes_tsv"), (PMM, "apply_wm_reref_selective"),
          (PERSP, "apply_grid_car")]),
        ("7 · trial tables and the trial filters",
         "Every <code>*.tsv</code> in <code>prep0/</code> is read (a byte-identical copy of an earlier file is skipped); it must have <code>sample</code>, <code>sample_offsets</code>, "
         "<code>trial_end</code>. Per condition a trial is dropped, first reason wins, if it touches a <code>bad_time_spans</code> stretch, if <code>resp_accuracy</code> is not one of "
         f"correct / valid / 1 (a missing column drops every trial), if the stimulus is shorter than {c.min_stim_s:g} s, or if the response (offset → trial end) is outside "
         f"{c.min_post_s:g}–{c.max_post_s:g} s; then the response durations of the survivors get Tukey fences (k = {c.iqr_k:g}) and what falls outside goes too. "
         "Writes <code>&lt;pid&gt;_IQR.tsv</code> and one duration figure per condition.",
         [(P140, None, "cond_groups = tr.collect_trials("), (PTR, "collect_trials")]),
        ("8 · per condition: the block, then the notch",
         f"Only for a patient in <code>notch_patients</code> (all of the current list). The signal is cut from {c.notch_block_pad_s:g} s before the first kept onset of the condition to {c.notch_block_pad_s:g} s after its last kept trial end. In that block, per shaft (<code>notch_shaft_all = {getattr(c, 'notch_shaft_all', False)}</code>; a shaft = "
         f"the name without its trailing number), the median Welch PSD (2 s segments) of the shaft's channels is tested at every multiple of {c.mains_base:g} Hz up to {c.fmax:g} Hz "
         f"(plus the combs of <code>notch_extra_bases</code>: {extra}): z = (peak within ±1 Hz − mean of the ring 1–5 Hz out) / SD of that ring, notched when z ≥ {c.notch_peak_z_thresh:g}. "
         f"Method <code>{c.notch_method_default}</code>: over the whole block's FFT the bins within the band get the RMS amplitude of the 2 Hz flanks on either side and a "
         f"<b>{c.notch_interp_phase}</b> phase (seeded). The half-width starts at the measured width (floors: {floors}), and grows by 1 Hz per pass — at most 12 passes — while the "
         f"3 Hz just outside stand more than 3 dB above the median 10–20 Hz away, up to <b>±{c.notch_interp_max_hw_hz:g} Hz</b>. Every candidate, notched or not, is a row of "
         "<code>&lt;pid&gt;_notch_audit.tsv</code> with its before / after level; peaks the combs do not explain go to <code>&lt;pid&gt;_unexplained_peaks.tsv</code>.",
         [(P140, None, "# ── PER-BLOCK CLEANING (notch_scope"), (PERSP, "block_window"), (PERSP, "apply_notch_with_audit"),
          (PERSP, "notch_per_shaft"), (PERSP, "notch_by_interpolation"), (PERSP, "peak_geometry"), (PERSP, "spectrum_interpolate"),
          (PERSP, "annotate_correction"), (PERSP, "unexplained_peaks_by_shaft")]),
        ("9 · per channel: the ERSP",
         f"Skipped for the reference contacts and, for MicroEPI with <code>--macros-only</code>, the micros. The channel is resampled to 1 kHz (polyphase), each trial is cut from "
         f"{c.baseline_w[0]:g} s before onset to its trial end, and its spectrogram taken: Hann, {c.nperseg} samples, overlap {c.noverlap} ({hop_ms:g} ms hop), nfft {c.nfft} "
         f"({df_hz:.3f} Hz bins), PSD density in dB, bins up to {c.fmax:g} Hz kept ({n_bins}). <b>Baseline</b>: per trial and per frequency, the mean of the dB frames whose centre "
         f"lies in [{c.baseline_calc_w[0]:g}, {c.baseline_calc_w[1]:g}) s is subtracted — a mean of dB values, not the dB of a mean power (if no frame falls in the window, the first tenth of the trial's frames is used instead, silently). <b>Time normalisation</b> (mode {c.mode}): "
         f"onset→offset is linearly interpolated onto bins 0–{n_stim - 1} and offset→trial end onto {n_stim}–{c.n_time_bins - 1}, each trial separately, so a bin is a fraction of that "
         "trial's own stimulus or response, not a latency. The trials are then averaged (nan-mean of dB).",
         [(P140, None, "for ci, chan_name in enumerate(names):"), (PERSP, "compute_ersp"), (PERSP, "_to_khz_resampled"), (PERSP, "_spectro")]),
        ("10 · inside the average: per-channel trial rejection",
         f"On the warped trials of the channel: score = 99th percentile of |dB| over 70–150 Hz; a trial is left out of the average and of both halves when its score exceeds "
         f"median + k × 1.4826 × MAD of that channel's scores, k = {rej.get('default', 'off')} for everyone (<code>ersp_trial_reject_hg_mad[\"default\"]</code>; a patient's own key wins). "
         "Fewer than 4 scored trials: no rejection. If more than 34 % would go, only the worst 34 % (ranked by the whole-map score) are dropped and the log says the channel may be bad. "
         "The rule is per channel: two channels of one patient can average different trials.",
         [(PERSP, "_reject_trials"), (PERSP, "_trial_scores"), (PERSP, "_rule_hits"), (P140, None, 'ersp_params.trial_reject_hg_mad = _rej("ersp_trial_reject_hg_mad")')]),
        ("11 · split halves",
         "The kept trials of the channel are split by position, even-indexed and odd-indexed, and each half averaged the same way. Saved without NaN filling.",
         [(PERSP, "_halves")]),
        ("12 · per-channel figures",
         f"<code>ERSP/</code>: the average. <code>PerTrial/HFA/</code>: every trial of the table, kept or not — the channel band-passed {c.hg_band[0]:g}–{c.hg_band[1]:g} Hz (Butterworth order 4, "
         f"zero-phase), Hilbert envelope, {c.hg_smooth_ms} ms boxcar, z-scored per trial against its own [{c.baseline_w[0]:g}, {c.baseline_w[1]:g}) s (a different window from the ERSP baseline); "
         "rows sorted by stimulus duration, excluded trials last with the reason, the per-trial z of the rejection score at the right. <code>PerTrial/Signal/</code>: the same rows as "
         f"traces of the cleaned broadband signal, each minus its own pre-stimulus mean, {getattr(c, 'signal_plot_uv_per_row', 200):g} µV per row and "
         f"{getattr(c, 'signal_plot_rows', 58)} rows in every figure; the signal is multiplied into µV when its robust SD says it is in volts. Not drawn for the photodiode.",
         [(PERSP, "plot_ersp"), (PERSP, "plot_hg_trials"), (PERSP, "plot_signal_trials"), (PERSP, "microvolt_scale"), (P140, "_hg_trial_z")]),
        ("13 · export: what stage 02 reads",
         "For a channel that is not non-neural, not bad-listed and not a reference contact: NaNs of the average are filled from the nearest valid bin, the cube is saved to "
         f"<code>ERSP_matrix/&lt;cond&gt;/</code> ({n_bins} × {c.n_time_bins} float), the two halves to <code>ERSP_halves/</code> (unfilled), and an axis-free image to <code>ERSP_clean/</code>. "
         "The file name carries the reference (<code>_WM_</code> / <code>_CAR_</code>).",
         [(P140, None, "if RUN_CLUSTER_EXPORT and chan_name not in skip:"), (PERSP, "fill_nans_nearest"), (PERSP, "save_clean_png")]),
        ("14 · per-trial scores and the run's reports",
         "<code>PerTrial/TrialScores/&lt;pid&gt;_&lt;cond&gt;_trial_scores.tsv</code>: one row per channel × trial that reached the ERSP — whole-map score, 70–150 Hz score, whether the rule dropped it, "
         "and the trial's number as the HFA figure prints it (non-neural channels have no rows). <code>PerTrial/Report/</code>: the notch audit and the unexplained peaks. "
         "The patient's row in <code>wm_reref_report.tsv</code> replaces its previous one; everything printed goes to <code>logs/&lt;pid&gt;_&lt;stamp&gt;.log</code>.",
         [(P140, None, "if EXPORT_TRIAL_SCORES and not _is_non_neural(chan_name):"), (P140, None, "# ── PER-BLOCK CLEANING REPORT"),
          (P140, "merge_report"), (P140, "main")]),
        ("after a run",
         "<code>141</code> reads the logs, the reports and the tree into one row per patient (<code>audit_140.tsv</code>). <code>142</code> lists, and with <code>--delete</code> removes, files older than "
         "their patient's run — 140 never clears a folder. <code>143</code> removes the files and score rows of non-neural channels from patients run before 2026-10-02 (keeps the "
         "photodiode's ERSP / HFA figures). <code>144</code> moves <code>HFA/</code>, <code>Signal/</code>, <code>TrialScores/</code>, <code>Report/</code> of patients run before the move into "
         "<code>PerTrial/</code>. <code>149</code> writes the Signal figures alone for a patient already in the tree (140's own <code>process_patient</code>, everything else switched off, in a scratch tree).",
         [("01_FBM_Analysis/141_audit_140.py", "main"), ("01_FBM_Analysis/142_stale_cubes.py", "main"),
          ("01_FBM_Analysis/143_delete_non_neural.py", "main"), ("01_FBM_Analysis/144_move_to_pertrial.py", "main"),
          ("01_FBM_Analysis/149_signal_plots_only.py", "main")]),
        ("the trial test tree",
         "<code>145</code> runs 140's <code>process_patient</code> into <code>04_ersp-trialtestzscore</code> with every rejection arm neutralised, so every trial is scored and none dropped. "
         "<code>146</code> z-scores each channel's scores across its trials and counts a trial as dropped when the share of channels over z reaches a fraction (a sweep; this rule is "
         "not in 140). <code>147</code> draws kept / removed trials and the spread of the trial z per patient; <code>148</code> the channel × trial z per patient, by shaft.",
         [("01_FBM_Analysis/145_trial_zscore_test.py", "main"), ("01_FBM_Analysis/146_trial_zscore_sweep.py", "per_channel_z"),
          ("01_FBM_Analysis/146_trial_zscore_sweep.py", "dropped"), ("01_FBM_Analysis/147_trial_figure.py", "draw"),
          ("01_FBM_Analysis/148_trial_z_heatmaps.py", "draw_patient")]),
        ("what reads the tree",
         "The cohort cache: <code>rebuild_concat_cache.py</code> → <code>lf_dataset.prepare_dataset</code> walks <code>&lt;tree&gt;/&lt;pid&gt;/LM/ERSP_matrix/&lt;cond&gt;/*.npy</code>. "
         "The LM visualizer's review mode: <code>make_lm_review_bundle.py</code> (cubes, QC names, audit row, trial tables; the three non-cohort patients from the frozen 04_* trees) and "
         "<code>make_lm_qc_images.py</code> (WebP copies of the HFA rasters, the trial-filter and PSD figures).",
         [("02_FBM_Clustering/rebuild_concat_cache.py", "main"), ("02_FBM_Clustering/functions/lf_dataset.py", "prepare_dataset"),
          ("02_FBM_Clustering/scripts/make_lm_review_bundle.py", "build_patient"), ("02_FBM_Clustering/scripts/make_lm_qc_images.py", "main")]),
    ]


def procedure_html() -> str:
    tr = []
    for title, text, cites in rows():
        cs = "<br>".join(cite(c)[0] for c in cites)
        tr.append(f'<tr><td style="white-space:nowrap"><b>{title}</b></td><td>{text}</td>'
                  f'<td style="font-size:11.5px;line-height:1.5">{cs}</td></tr>')
    return ('<div style="overflow-x:auto"><table class="cmp" style="font-size:12.6px">'
            '<tr><th>step</th><th>what the code does</th><th>where &mdash; path from <code>Analysis_LoraFanda/</code> &middot; function &middot; line</th></tr>'
            + "".join(tr) + "</table></div>")


def main() -> int:
    n = 0
    for title, text, cites in rows():
        print(html.unescape(title.replace("<code>", "").replace("</code>", "")))
        for c in cites:
            print("    " + cite(c)[1]); n += 1
    print(f"\n{n} citations resolved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
