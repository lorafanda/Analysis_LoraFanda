# Notebook 13 — `13_MicroSpikeSorting_Full.ipynb` — Walkthrough & Reference

*Micro-electrode (Blackrock NS6, 30 kHz) spike detection for the MicroEPI language-mapping
task. Diagnostic pipeline: estimate, per tetrode, how many candidate units exist and
when they fire. Clustering/sorting proper is out of scope (left to Combinato / Wave_clus
or a future notebook 14).*

> This document is written to be shared with a colleague who has never opened the
> notebook. It covers (1) what the pipeline does end-to-end, (2) a cell-by-cell guide,
> (3) the signal-processing algorithms it relies on, (4) the current state of the code
> and the gotchas you must know before running it, and (5) the roadmap for the two
> requested improvements (per-channel adaptive notch, per-contact spike sorting).

---

## 0. TL;DR — the one-paragraph version

Six NS6 fragments are stitched into one continuous 30 kHz stream. Unused "X" channels are
dropped. An **adaptive 50 Hz mains-harmonic notch** is applied to the micro channels to
remove line noise and its harmonics. Canonical trial timing (stim on/off, trial end, per
condition) is recovered from the photodiode + behavioural TSV using the *same* code path as
the ERSP/HG notebook 11. Micros are grouped into **tetrodes** (4 wires each). For each
tetrode: bandpass 300–6000 Hz → intra-tetrode common-average reference (CAR) → MAD-based
threshold detection (per wire, merged across wires with a refractory window) → snippet
extraction, with saturation- and fragment-boundary guards. Spikes are then characterised
(rate-over-time, waveform templates, ISI/contamination, peak-wire distribution, PSTH locked
to stim) and cached to an `.h5`. A second half of the notebook re-loads that cache and runs
heavier diagnostics + optional strict quality filtering.

---

## 1. Pipeline at a glance

```
NS6 fragments (×6, 30 kHz)
   └─ read_blackrock_session ──────────────► signals_micro (samples × micro-chans)
                                             signals_ainp  (analog inputs incl. photodiode)
                                             fragment_boundaries_s
   └─ (optional) keep only KEEP_PREFIXES tetrodes
   └─ notch_mains_harmonics  ──────────────► signals_micro (line-noise cleaned)   ★ NOTCH
   └─ PSD QC (before vs after notch)         [diagnostic plot + dB residual table]

photodiode (ainp2) + behavioural .tsv
   └─ extract_trials_with_qc ──────────────► cond_groups[cond] = (ons, offs, tends)  [sample units]
                                             on_abs, off_abs

per tetrode (4 wires):
   raw → bandpass_spike_band(300–6000) → intra_tetrode_car → detect_threshold_crossings
       → saturation guard (>MAX_AMPLITUDE_UV) → fragment-boundary guard (±FRAGMENT_GUARD_MS)
       → extract_snippets  ─────────────────► spikes_per_tetrode[label] = {spike_times_s,
                                               snippets, peak_channel, thresholds, wires…}

diagnostics → PSTH (locked to LOCK_TO) → save .h5 cache

[second half] reload .h5 → enhanced diagnostics → (optional) restrict-to-trial-window
            → strict quality filter → re-plot
```

`★ NOTCH` is the stage targeted for the per-channel improvement.

---

## 2. Environment & dependencies — **read this first**

The notebook imports four local helper modules from `01_FBM_Analysis/functions/`
(`sys.path.insert(0, "functions")`):

| import | alias | what it provides |
|---|---|---|
| `lf_blackrock_io` | `bb` | `read_blackrock_session`, `extract_analog_channel`, `read_ns6_fragment`, `detect_photodiode_onsets` |
| `lf_spikes` | `sp` | `bandpass_spike_band`, `intra_tetrode_car`, `detect_threshold_crossings`, `extract_snippets`, `align_snippets_by_trough`, `plot_spike_rate_over_time`, `plot_snippet_overlay`, `plot_psth`, `plot_raster` |
| `lf_micromacro` | `mm` | `notch_mains_harmonics` (the notch), `extract_trials_with_qc`, … |
| `config` | — | `MICROEPI_PRESETS` |

### ⚠️ Critical: the source code for the spike/IO modules and the notch is **missing from disk**

As of this commit, `functions/` contains **only** `config.py`, `lf_ersp.py`,
`lf_io_utils.py`, `lf_micromacro.py`, `lf_recon.py`, `lf_trials.py`. The modules the
notebook actually needs to run are **gone**:

- `functions/lf_spikes.py` — **does not exist** (only `__pycache__/lf_spikes.cpython-311.pyc` remains).
- `functions/lf_blackrock_io.py` — **does not exist** (only the `.pyc` and a stale `.ipynb_checkpoints` copy remain).
- The current `functions/lf_micromacro.py` (529 lines) **does not contain `notch_mains_harmonics`** (nor `extract_trials_with_qc`). The version that did is only present as `__pycache__/lf_micromacro.cpython-311.pyc` (≈100 KB, compiled from a much larger source that has since been overwritten).

None of these are tracked in git history either. **Consequence: the notebook cannot be
re-run as-is against the current working tree** — the imports in Cell 1 will fail (or, for
`mm`, the `mm.notch_mains_harmonics` / `mm.extract_trials_with_qc` calls will raise
`AttributeError`). The stored cell outputs are from a previous run under **Python 3.11**
(the `.pyc` are `cpython-311`); the machine's default `python` is 3.6.

The exact signatures and algorithms of the missing functions have been recovered from the
compiled `.pyc` and are documented in §4 below, so the modules can be restored faithfully.

### ✅ Restored (2026-06) — how the notebook runs again

The missing functionality was replaced with **readable source** wherever an existing `.py`
made that possible, leaving only one unavoidable compiled module:

**New / readable source modules** (preferred — fully inspectable, no version lock):
- `functions/lf_micro_io.py` — NS6 session loader (`read_blackrock_session`, `read_ns6_fragment`,
  `extract_analog_channel`). Reads via `neo.io.BlackrockIO` and **rescales to microvolts**, the
  same approach the deleted `lf_blackrock_io` used. (Note: `lf_io_utils.load_ns6` uses
  `BlackrockRawIO`/`get_analogsignal_chunk`, which returns *raw int16 ADC counts* — building on
  that would silently break the µV amplitude knobs, so `lf_micro_io` reads via `BlackrockIO`.)
  → replaces `lf_blackrock_io.pyc`, which was deleted.
- `functions/lf_notch.py` — per-channel adaptive notch (§7.1).
- `functions/lf_sort.py` — configurable per-contact/tetrode detection (§7.2) **plus** the
  diagnostic plot helpers (`plot_spike_rate_over_time` / `plot_snippet_overlay` / `plot_psth`)
  → replaces `lf_spikes.pyc`, which was deleted.

- `functions/lf_trials_qc.py` — readable reimplementation of `extract_trials_with_qc` (the
  canonical photodiode+TSV trial timing + per-condition hard/IQR QC). The heavy lifting
  (photodiode detection, trial pairing, fake/invalid handling, trial-end, the QC plot) was
  never lost — it lives in `LFfunctions_PDextract.get_trigger_indexes_photodiode`, which this
  module calls; only the thin QC/grouping wrapper was rebuilt. → replaces
  `lf_micromacro_full.pyc`, which was deleted. (The current `lf_micromacro.py` is a different,
  unrelated module and is untouched.)

**No compiled modules remain** — the entire pipeline is now plain, inspectable `.py`, with no
Python-version lock.

**Notebook rewiring** (Cells 2–11): imports now use `bb=lf_micro_io`, `nf=lf_notch`,
`ls=lf_sort`, `mmf=lf_trials_qc`; `lf_spikes` is gone (plots come from `ls`). Cell 2 (config) per-channel
notch + `KEEP_PREFIXES` tuple fix; trial-timing cell = `mmf.extract_trials_with_qc`; detection
cell corruption removed → `ls.sort_session(detection_unit="wire")`; diagnostics plots → `ls.plot_*`;
config import aliased (`MICROEPI_MAT_PRESETS as MICROEPI_PRESETS`). A cell after the h5 save
exports the FBM binary (see §8).

> ⚠️ **Validation status.** All modules import, every code cell parses (0 syntax errors), the
> notch + detection algorithms are unit-tested, the trial-QC helpers (`_iqr_keep` etc.) are
> spot-checked, and all external call sites bind against the real signatures. The one piece with
> no ground-truth-to-date is the trial-QC wrapper: confirm the per-condition trial counts look
> right on your first run (the preset declares `trial_ids`, so the kept counts should be sane).
> A full data run (needs the NAS raw NS6 + the lab's normal scientific stack incl. `neo`) was
> not executed here.

### Outputs / GitHub hygiene

The repo `.gitignore` already excludes `outputs/`, `*.h5`, `__pycache__/`, and `.ipynb_checkpoints/`,
and a `.githooks/pre-commit` bloat guard blocks any single staged file > 5 MB. The notebook's
heavy embedded image outputs were stripped (15.7 MB → ~90 KB) so it is committable and carries
no heavy outputs to GitHub — re-running regenerates the figures locally. The FBM `.bin` export
(§8) and the `.h5` cache are written under `outputs/` and are therefore never pushed.

### 8. FBM binary export (`FBMdata_DD-MM-YY.bin`)

The cell after the `.h5` save writes the whole cleaned micro recording as a flat **int16**
binary — the standard "data in the end" for downstream spike sorters (Combinato / Kilosort):

- `outputs/13_microSpikeSorting/<pid>/FBMdata_DD-MM-YY.bin` — int16, little-endian, C-order
  `(n_samples, n_channels)`, the **notched wideband** `signals_micro`. Date parsed from
  `base_filename` (e.g. `20260128…` → `28-01-26`).
- `…/FBMdata_DD-MM-YY.json` — sidecar: `fs`, channel names/order, shape, dtype, `units=microvolts`,
  `gain_uV_per_unit` (default 1.0 → values are rounded µV), fragment boundaries, clip count.
- Reload: `np.fromfile(path, dtype='<i2').reshape(n_samples, n_channels)`.
- Exports the **current** `signals_micro` (the `KEEP_PREFIXES` subset if set) — set
  `KEEP_PREFIXES = None` in Cell 2 for the whole probe. `FBM_BIN_GAIN` lets you trade µV
  resolution vs dynamic range if clipping is reported.

---

## 3. Cell-by-cell guide

| Cell | Type | Purpose | Notes / gotchas |
|---|---|---|---|
| 0 | md | Title + pipeline summary | Accurate description of intent. |
| 1 | code | **Imports + per-patient config** | All tunable knobs live here. Patient = `MicroEPI-G-06`, `fs=30000`. `LOCK_TO="stim_off"` controls cells 6 & 8. `K_MAD=6.0`, `SPIKE_POLARITY="neg"`, `MAX_AMPLITUDE_UV=300`. `SERVER_ROOT` is a TODO placeholder you must set. |
| 2 | code | **Load NS6 + subset + notch** | `KEEP_PREFIXES=("ADm")` ⚠️ this is a *string*, not a tuple — `any(n.startswith(p) for p in "ADm")` iterates the characters `'A','D','m'`, so the subset filter is broader than it looks (keeps anything starting with A, D, or m). Use `("ADm",)`. Calls `mm.notch_mains_harmonics` on **all kept micro channels at once** (this is the per-group notch). |
| 3 | code | **PSD QC: before vs after notch** | Welch PSD, channel-mean, log-log plot with 50 Hz harmonic grid; prints residual harmonic peak-vs-floor in dB (want post ≈ 0). This is the figure you shared. `figsize=(100,4)` is intentionally ultra-wide. |
| 4 | code | **Trial timing (canonical)** | Builds a minimal `d_all`, sets `preset["trig"]=photodiode_channel`, `time_range=(50,1444)`, runs `mm.extract_trials_with_qc`. Produces `cond_groups`, `on_abs`, `off_abs`. Plots PD trace + onsets/offsets for first 60 s. |
| 5 | code | **Per-tetrode spike detection** | The core detection loop (group→bandpass→CAR→detect→saturation guard→fragment guard→snippets) writing `spikes_per_tetrode`. ⚠️ **This cell is corrupted** — a broken-indentation *duplicate* of the Cell-13 diagnostics block has been pasted onto the end of the final `print(...)` (note the stray `a` and a trailing `"`). As written it raises a `SyntaxError`; the stored output is from before the corruption. The detection logic above the corruption is correct and is what should be kept. |
| 6 | code | **Per-tetrode diagnostics + anchor selection** | Rate-over-time, snippet overlay, ISI histogram (2 ms refractory metric), peak-wire bar. Defines `get_anchor_samples` (maps `LOCK_TO`→sample anchors) and prints a per-tetrode spikes-per-trial table over `TRIAL_WINDOW_S=(0,1)`. |
| 7 | code | (blank placeholder) | Intentionally empty. |
| 8 | code | **Per-condition PSTH** | `n_tet × n_cond` grid, `sp.plot_psth`, window `(-1, 3)` s, 0.1 s bins, locked to `LOCK_TO`. |
| 9 | code | **Save `.h5` cache** | Writes `session/`, `trials/` (incl. `cond_groups`), `spikes/` (per-tetrode `spike_times_s`, `snippets`, `peak_channel`, `thresholds`, wire metadata). Snippets stored on disk as **(N, time, wires)**. Path: `outputs/13_microSpikeSorting/<pid>/<pid>_spikes.h5`. |
| 10 | md | "IF PREPED DATA:" — divider for the reload half. | |
| 11 | code | **Reload cache (variant A)** | ⚠️ Uses the *wrong* path `outputs/{pid}_spikes.h5` (flat) — superseded by Cell 12. Treat as dead. Transposes snippets back to **(N, wires, time)**. |
| 12 | code | **Reload cache (variant B)** | Correct nested path. This is the loader to use. |
| 13 | code | **Enhanced diagnostics (5-col)** | Rate, feature scatter (peak-amp vs width, KMeans for *visualisation only*), template+SNR, ISI+contamination, wire distribution. Prints a quality summary. |
| 14 | code | **Enhanced diagnostics + noise recs (6-col)** | Adds waveform overlay + quality-score colouring; prints per-tetrode noise status + suggested parameter changes. |
| 15 | code | **Fine-grained rate vs stim overlay** | 1 s bins, photodiode panel on top, condition-coloured stim spans. `figsize` very wide. Falls back to `.h5` for timing if not in scope. |
| 16 | code | **Restrict spikes to in-trial windows** | Builds the union of `[on−PRE_PAD, tend+POST_PAD]` intervals, merges them, filters `spikes_per_tetrode` **in place**. Run after the loader, before the strict filter. `RESTRICT_TO_TRIAL_WINDOWS=True`. |
| 17 | code | **Strict quality filter** | SNR / waveform-shape / amplitude-bounds / spatial-consistency / subsample, in place. ⚠️ `SNR_MIN = .0` → the SNR filter is effectively **disabled** despite the "strict" label. Amplitude band 40–800 µV, `MIN_SPATIAL_CONSISTENCY=60%`, trough-to-peak 0.25–0.8 ms. |
| 18 | code | **Re-run of Cell 14** | Identical diagnostics, re-plotted after filtering (A/B comparison). |
| 19 | code | **Re-run of Cell 15** | Identical rate-vs-stim, after filtering. |
| 20 | code | empty | |

**Duplication summary:** 13≈(first half of)5-corruption, 14≡18, 15≡19. The second occurrences
exist so you can see diagnostics *after* the restrict (16) + strict-filter (17) steps.

---

## 4. The algorithms (reconstructed from the compiled modules)

### 4.1 `notch_mains_harmonics` — the adaptive mains notch  ★ target for improvement

Recovered signature & defaults:

```python
notch_mains_harmonics(X, fs, *, base=50.0, max_hz=None, repeats=1, peak_z_thresh=3.0,
                      Q_min=10.0, Q_max=400.0, local_hz=5.0, peak_hz=1.0,
                      min_df=0.3, max_df=1.5, df_safety=1, verbose=True)
```

Reconstructed logic (faithful to the bytecode):

```python
from scipy.signal import welch, iirnotch, filtfilt
X = np.asarray(X, float)
was_1d = X.ndim == 1
if was_1d: X = X[:, None]
nyq = 0.5 * fs
lim = min(0.98*nyq, max_hz if max_hz is not None else 0.98*nyq)
harms = [h*base for h in range(1, int(lim // base) + 1)]   # 50,100,150,… Hz
nper = int(4.0 * fs)                                        # 4-s Welch segments

# ---- PSD used to find peaks: the CHANNEL-MEDIAN over ALL channels passed in ----
if X.shape[1] == 1:
    f_psd, P_med = welch(X[:,0], fs=fs, nperseg=nper, detrend=False)
else:
    P_all = [welch(X[:,ci], fs=fs, nperseg=nper, detrend=False)[1]
             for ci in range(X.shape[1])]
    f_psd = welch(X[:,0], fs=fs, nperseg=nper, detrend=False)[0]
    P_med = np.median(np.vstack(P_all), axis=0)            # ← one PSD for the whole group

P_db   = 10*np.log10(P_med + 1e-30)
df_bin = float(f_psd[1] - f_psd[0])
Y = X.copy()

for f0 in harms:
    if f0 >= 0.95*nyq: continue
    local  = (f_psd >= f0-local_hz) & (f_psd <= f0+local_hz)   # ±5 Hz window
    peak_w = (f_psd >= f0-peak_hz ) & (f_psd <= f0+peak_hz )   # ±1 Hz peak band
    bg = local & ~peak_w
    if not (np.any(bg) and np.any(peak_w)): continue
    bg_mean = float(np.mean(P_db[bg]))
    bg_std  = float(np.std (P_db[bg]) + 1e-6)
    local_idx = np.flatnonzero(local)
    pk_idx = local_idx[np.argmax(P_db[local])]
    pk_val = float(P_db[pk_idx])
    z = (pk_val - bg_mean) / bg_std
    if z < peak_z_thresh:                       # no real peak here → don't notch
        continue
    # FWHM via half-power (−3 dB) walk out from the peak
    half = pk_val - 3.0
    i = pk_idx
    while i > local_idx[0]  and P_db[i] > half: i -= 1
    j = pk_idx
    while j < local_idx[-1] and P_db[j] > half: j += 1
    df = max(f_psd[j]-f_psd[i], 2*df_bin)
    df = float(np.clip(df, min_df, max_df))     # FWHM clamp 0.3–1.5 Hz
    Q  = float(np.clip(f0 / (df_safety*df), Q_min, Q_max))   # auto-Q, clamp 10–400
    b, a = iirnotch(w0=f0, Q=Q, fs=fs)
    for _ in range(int(repeats)):
        Y = filtfilt(b, a, Y, axis=0, method='gust')   # zero-phase, all channels at once
return Y.squeeze() if was_1d else Y
```

**Key properties**

- **Adaptive in frequency:** a harmonic is only notched if a real spectral peak is present
  (z-score of the peak over the local ±5 Hz background ≥ `peak_z_thresh`). Quiet harmonics
  are skipped, so no needless signal distortion.
- **Adaptive in width:** the notch `Q` is derived from the measured half-power FWHM of each
  peak (`Q = f0 / (df_safety·FWHM)`), clamped to `[Q_min, Q_max]`. Wider peaks → lower Q →
  wider notch.
- **Zero-phase:** `filtfilt(..., method='gust')`, so no phase distortion of spike waveforms.
- **★ The per-group limitation (what you want to change):** the peak detection runs on a
  **single channel-median PSD** computed across *every channel passed in*. In Cell 2 that
  is the entire `signals_micro` array (all kept micro contacts across all tetrodes). The
  *same* set of harmonics and the *same* per-harmonic `Q` are then applied to *all* channels.
  If one contact has a strong 350 Hz harmonic that is weak on the others, the median washes
  it out and that contact never gets notched there; conversely a peak present on only a few
  contacts can drive a notch that is applied to channels that didn't need it. This is the
  root cause of residual line noise leaking into individual micro channels.

### 4.2 Spike-band conditioning & detection (`lf_spikes`)

Recovered signatures:

```python
bandpass_spike_band(signals, fs, *, low=300.0, high=6000.0, order=4)   # Butterworth, zero-phase
intra_tetrode_car(signals_tetrode)                                     # X - mean over the 4 wires
detect_threshold_crossings(signals_filt, fs, *, k_mad=4.5, polarity='neg',
                           refractory_ms=1.0, verbose=True)
extract_snippets(signals_filt, spike_idx, fs, *, pre_ms=0.5, post_ms=1.5)
align_snippets_by_trough(snippets, *, search_window=None)
_mad_sigma(x)   # Quian-Quiroga robust noise: sigma = median(|x|)/0.6745
```

- **`bandpass_spike_band`** — zero-phase Butterworth band 300–6000 Hz (high edge clamped to
  0.95·Nyquist), returns float32.
- **`intra_tetrode_car`** — subtracts the across-4-wire mean from each wire. Removes shared
  (common-mode) noise without cancelling true single-unit signals (which are *not* common
  across the bundle). Requires `(n_samples, n_wires)`.
- **`detect_threshold_crossings`** — robust MAD detector:
  - Per channel: `sigma = median(|x|)/0.6745` (`_mad_sigma`), `threshold = k_mad·sigma`.
  - Detects threshold crossings **on each wire independently** (`<−thr` for `'neg'`,
    `>thr` for `'pos'`, either for `'both'`).
  - **Merges across wires** with a refractory window (`refractory_ms`): events closer than
    the window are collapsed to a single spike, and `peak_channel` is set to the wire with
    the largest deflection. → **one event per tetrode-neuron, deduplicated across wires.**
  - Returns `spike_idx (int64)`, `peak_channel (int8)`, `thresholds (per-channel float)`.
- **`extract_snippets`** — cuts `pre_ms`/`post_ms` windows around each spike across all wires;
  drops events whose window would run off the recording edge (returns a `keep_edge` mask).
  Shape **(N, time, wires)** as written to `.h5`; the loader transposes to **(N, wires, time)**.

The notebook tightens the library defaults: `K_MAD=6.0` (vs 4.5), `REFRACTORY_MS=1.5` (vs 1.0),
plus a `MAX_AMPLITUDE_UV=300` saturation guard and a `±100 ms` fragment-boundary guard that
the library functions do not impose themselves.

### 4.3 IO (`lf_blackrock_io`)

```python
read_ns6_fragment(ns6_path, *, verbose=True)
read_blackrock_session(data_dir, base_filename, file_suffixes, *,
                       micro_channel_pattern=None, drop_micro_pattern=r'^[Xx](_?\d+)?$', verbose=True)
extract_analog_channel(d_session, channel_name)
detect_photodiode_onsets(pd_signal, fs, *, threshold=None, threshold_frac=0.5,
                         min_gap_s=0.3, polarity='auto', verbose=True)
```

`read_blackrock_session` stitches the six NS6 fragments into one stream and returns the
session dict (`fs`, `signals_micro`, `signals_ainp`, `names_micro`, `names_ainp`,
`duration_s`, `fragment_boundaries_s`). Note the notebook overrides `drop_micro_pattern`
with `r"^[Xx]\d+[mM]\d+$"`.

---

## 5. Outputs

- **Figures** (inline): PSD before/after notch (the harmonics figure), PD onset/offset QC,
  per-tetrode diagnostics (rate/snippets/ISI/peak-wire), PSTH grid, rate-vs-stim with PD
  overlay, enhanced 5/6-column diagnostics.
- **`.h5` cache:** `outputs/13_microSpikeSorting/MicroEPI-G-06/MicroEPI-G-06_spikes.h5`
  containing `session/`, `trials/` (with `cond_groups`), and `spikes/` per tetrode.
- **QC TSVs** written by `extract_trials_with_qc` into `qc_dir`.

---

## 6. Current state & gotchas (the must-knows)

1. **Missing modules** (§2): `lf_spikes.py`, `lf_blackrock_io.py`, and the notch-bearing
   `lf_micromacro.py` are not on disk — restore them from the `.pyc`/reconstructions before
   anything will run.
2. **Cell 5 is corrupted** — a mangled duplicate diagnostics block is appended after the
   detection loop; it must be removed (keep only the detection loop).
3. **`KEEP_PREFIXES=("ADm")` is a string, not a tuple** — broadens the channel subset
   unintentionally. Use `("ADm",)`.
4. **`SNR_MIN = .0` in Cell 17** — the "strict" SNR filter is a no-op as written.
5. **Cell 11 uses the wrong cache path** — use Cell 12.
6. **Python version:** outputs were produced under 3.11; the default interpreter here is 3.6.
   Run under the 3.11 that produced the `.pyc`.
7. **Units:** all trial timing (`ons/offs/tends`, `on_abs/off_abs`) is in **samples** of
   `fs=30000`. Spike times are in **seconds**.

---

## 7. Roadmap for the two requested improvements

### 7.1 Per-channel adaptive notch — **IMPLEMENTED** in `functions/lf_notch.py` ✅

Status: done and numerically verified. `lf_notch.notch_mains_harmonics(X, fs, *,
mode="per_channel", ...)` detects harmonic peaks and derives each notch `Q` on **each
channel's own PSD**, and notches each channel only with the harmonics that are real on *that*
channel. `mode="median"` preserves the legacy per-group behaviour for A/B comparison.
Improvements over the legacy version: FWHM-based auto-`Q` (notch width tracks peak width),
median-averaged Welch PSD (robust to transient artifacts), a per-channel audit, NaN/flat-channel
guards, and a `harmonic_residual_db` QC helper for per-channel before/after dB.

Verification (synthetic 30 kHz, 6 ch; common 50/100/150 Hz on all, a 350 Hz harmonic on **one**
channel only, plus a 222 Hz signal-of-interest): the legacy median mode left +37 dB residual at
350 Hz on the carrying channel; per-channel mode reduced it to ~0 dB, did **not** place a notch
on channels lacking it, and preserved the 222 Hz tone to within 0.000 dB. All assertions passed.

Wiring into the notebook (Cell 2): `import lf_notch as nf` and replace
`mm.notch_mains_harmonics(signals_micro, fs, ...)` with
`nf.notch_mains_harmonics(signals_micro, fs, base=NOTCH_BASE_HZ, max_hz=min(NOTCH_FMAX_HZ,0.5*fs),
repeats=NOTCH_REPEATS, peak_z_thresh=NOTCH_Z_THRESH, mode="per_channel", return_audit=True)`.
Cell 3 QC can use `nf.harmonic_residual_db(...)` to report per-channel (not just channel-mean)
residuals.

Original design sketch (now realised):

- Refactor `notch_mains_harmonics` into (a) a per-channel core
  `_notch_one_channel(x, fs, …)` that runs the existing z-test + FWHM-Q logic on that
  channel's own Welch PSD, and (b) a vectorised wrapper that loops channels (or processes
  them in parallel) and returns the cleaned matrix.
- Keep the same knobs (`peak_z_thresh`, `Q_min/Q_max`, `local_hz`, `peak_hz`, `min_df/max_df`,
  `df_safety`, `repeats`) so results stay comparable; add a `mode={"per_channel","median"}`
  switch defaulting to `per_channel`, so A/B comparison (and the existing behaviour) is one
  flag away.
- Robustness to add: optionally require a peak to persist across `repeats` independent PSD
  estimates; report a per-channel audit (which harmonics notched, z, FWHM, Q) so the PSD-QC
  cell can show before/after **per channel**, not just the channel mean.
- Validate with the Cell-3 residual-dB table extended to per-channel worst-case, not mean.

### 7.2 Spike sorting on each contact of each tetrode — **IMPLEMENTED** in `functions/lf_sort.py` ✅

Status: done and verified. `lf_sort.py` is a clean standalone reconstruction of the spike
helpers (`bandpass_spike_band`, `intra_tetrode_car`, MAD `detect_threshold_crossings`,
`extract_snippets`, `group_micros_into_tetrodes`) plus a high-level `sort_session(...)` driver,
with the new **configurable detection unit**:

- `detection_unit="tetrode"` — reproduces the original behaviour exactly (one event per
  neuron, merged across the 4 wires, `peak_channel` = max-deflection wire).
- `detection_unit="wire"` — detects on **each contact independently** (no cross-wire merge);
  a spike crossing several wires becomes one event per wire. The driver returns one entry per
  contact, keyed `"<tetrode>::<wire name>"`, giving per-contact spike trains + snippets.

The tetrode merge is the exact greedy refractory algorithm recovered from the original
`lf_spikes` bytecode; the `group_micros_into_tetrodes` here fixes the notebook's
`KEEP_PREFIXES`/exclude bug. Verified on synthetic tetrode data (10 assertions): a 3-wire
coincident event → 1 tetrode event on the correct peak wire / 3 per-contact events; a
wire-private spike stays on its own contact; a per-contact train is counted correctly;
refractory + edge guards hold; intra-tetrode CAR cancels common-mode (|mean| 40 → 0 µV).

Wiring into the notebook (replaces the Cell-5 detection loop):
`import lf_sort as ls; spikes = ls.sort_session(signals_micro, names_micro, fs,
detection_unit="wire", k_mad=K_MAD, polarity=SPIKE_POLARITY, refractory_ms=REFRACTORY_MS,
max_amplitude_uv=MAX_AMPLITUDE_UV, fragment_boundaries_s=fragment_boundaries_s,
fragment_guard_ms=FRAGMENT_GUARD_MS)`. The returned dict is shape-compatible with
`spikes_per_tetrode` (per-contact keys in wire mode), so the existing diagnostics cells work
with minimal change.

Original design options (resolved -> "configurable both"):

- **(a) Per-wire detection** — run `detect_threshold_crossings` on each single wire (no
  cross-wire merge), yielding independent spike trains per contact. Simplest interpretation
  of "each contact".
- **(b) Tetrode detection, per-contact features** — keep the merged tetrode events (better
  for true tetrode sorting) but expose per-contact waveforms/SNR/threshold and split
  diagnostics by `peak_channel`.
- **(c) Both / configurable** — a `detection_unit={"wire","tetrode"}` switch.

This choice changes the data model (`spikes_per_contact` vs `spikes_per_tetrode`) and the
`.h5` schema, so it's worth aligning before implementing.

---

*Generated from a read of the notebook and a reconstruction of the compiled helper modules
(`__pycache__/*.cpython-311.pyc`). Algorithms in §4 are faithful to the bytecode; restore the
`.py` sources before relying on them in production.*
