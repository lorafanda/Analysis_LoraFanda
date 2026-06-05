How to run this notebook:


1. 140 (preprocess)
2. 251 cell 2 + cell 4 (recon)
3. 210 (one clustering)
4. 213 (cluster ranking)
5. 252 (recon join)
6. git push + hard-refresh MOBA

# Clustering pipeline — what each notebook does

This document explains the role of every notebook in the FBM clustering pipeline,
in execution order. Two upstream notebooks live in `01_FBM_Analysis/` and produce
the inputs everything else consumes; the rest live in `02_FBM_Clustering/`.

---

## Pipeline overview

```
01_FBM_Analysis/140 → produces ERSP_matrix/*.npy per electrode
                          │
                          ▼
02_FBM_Clustering/210 ─────┐
02_FBM_Clustering/212 ─────┤
02_FBM_Clustering/230 ─────┼─→  cluster each sample in different feature spaces
02_FBM_Clustering/231 ─────┤    → outputs/clustering/{method}/{feature_set}/runs/<id>/
02_FBM_Clustering/232 ─────┘
                          │
                          ▼
02_FBM_Clustering/211 → post-hoc validation (stability + anatomy + gap stat)
02_FBM_Clustering/252 → 3D fsaverage brain recon per cluster
                          │
                          ▼
                       MOBA dashboard (lorafanda.github.io/moba.html)
```

All five clustering notebooks (`210`, `212`, `230`, `231`, `232`) use the **same canonical
sample set** — built once by `functions.lf_dataset.prepare_dataset` (high-activity-gated
ERSPs from `04_ersp_LM_RAWONLY/`) and cached in `outputs/_dataset/canonical/`. This is what
makes cross-feature-set comparisons valid: every method clusters the same ~1500 samples.

---

## 140 — `01_FBM_Analysis/140_ERSP_analysis_pipeline.ipynb`

**Role**: upstream ERSP extraction. Produces the `.npy` files every other notebook reads.

- Loads raw iEEG signals per patient (TRC / EDF / H5 / MicroEPI .mat)
- Filters out non-neural channels (PHOTO, X*, E*, ECG, AUDIO, +/− markers)
- For EL patients: keeps only channels with anatomical prefixes
- **Drops `Unknown` parcellation channels** (read from the BIDS electrodes TSV — never enter ERSP compute)
- **WM rereferencing** via `apply_wm_reref` (or pass-through with `reref='NONE'` for grid patients like EL044)
- Adaptive notch filtering at mains harmonics
- Loads photodiode triggers → trial onsets / offsets (IQR-based outlier rejection)
- For each (patient × condition × channel × trial): computes ERSP via STFT + warps to 50% stim / 50% post-stim
- Saves per-channel outputs:
  - `outputs/04_ersp_LM_RAWONLY/<pid>/LM/ERSP_matrix/<cond>/<stem>.npy` — the dB-normalized 129×300 ERSP matrix
  - `outputs/04_ersp_LM_RAWONLY/<pid>/LM/ERSP_clean/<cond>/<stem>_CLEAN.png` — display PNG for MOBA
  - `outputs/04_ersp_LM_RAWONLY/<pid>/LM/HG/<cond>/<stem>.png` — high-gamma waterfall plot
  - `outputs/04_ersp_LM_RAWONLY/<pid>/LM/PSD_raw/` + `PSD_clean/` — QC PSD plots
- **End-of-batch WM report**: prints a per-patient table (channels in / neural / unknown_dropped / used / WM channels used) and saves `wm_reref_report.tsv`
- Memory hygiene: each patient processed inside `process_patient(pid)`, then `gc.collect()` + `plt.close('all')` between patients

**Run this first whenever you change patients, conditions, or any signal-processing parameter.**

---

## 210 — `02_FBM_Clustering/210_raw_clustering.ipynb`

**Role**: cluster on the **full-resolution raw ERSP** (129 freq × 300 time = 38,400 features).

- Loads the canonical sample set via `prepare_dataset` (high-activity-gated)
- Stacks ERSPs → `X_3d` shape `(n_samples, N_FREQ, N_TIME)`
- Flattens → `X_raw` shape `(n_samples, 38,400)` → standardized → `X_scaled`
- Calls `R.fit_and_save` twice:
  - **K-Means K-sweep** over `K_RANGE = [10..20]`, `n_init=20`, picks best K by silhouette
  - **Hierarchical (Ward)** K-sweep over the same range
- Writes per-run artifacts to `outputs/clustering/{kmeans,hierarchical}/raw/runs/<id>/`:
  - `manifest.json`, `metrics.json`, `labels.csv` (with per-sample silhouette), `model.joblib`, `predictor.joblib`, `X_train.npy`
  - `cluster_labels_by_k.csv` — wide format, K columns 10..20 (powers MOBA's K-slider)
  - `silhouette_by_k.{json,png}`, `centroids.png`, `silhouette_per_cluster.png`, `similarity_heatmap.png`, `centroid_distance_heatmap.png`
- **`_ersp_centroid_grid` cell**: large grid of per-cluster mean ERSPs (one panel per cluster, full 129×300)
- **`BACKFILL_CENTROIDS` cell**: tiny per-cluster thumbnail PNGs (`cluster_centroids/cluster_NN.png`) for the MOBA cluster chips

**Limitation**: 38,400-feature clustering is curse-of-dimensionality territory. Silhouettes hover at ~0.05. **Use 212 instead for the paper.**

---

## 212 — `02_FBM_Clustering/212_raw_downsampled_clustering.ipynb`

**Role**: cluster on a **band-aware downsampled raw ERSP** (15 freq bands × 30 time bins = 450 features). Same sample set as 210, much lower dimensional, much higher silhouettes.

- Loads canonical sample set (cache hit, instant)
- For each ERSP, calls `lf_features.downsample_ersp_to_bands` to compress
  - **Frequency axis**: average input bins by neuroscience-tailored bands (delta / theta / alpha / low+high beta / low+mid gamma / 3× HG sub-bands / 5× HFO sub-bands up to 400 Hz). Default `FREQ_BANDS_15_TO_400HZ` in `lf_features.py`.
  - **Time axis**: 300 → 30 via skimage anti-aliased resize (same convention as minus101)
- Stacks → `X_3d_ds` shape `(n, 15, 30)` → flatten → `X_raw` → `StandardScaler` → `X_scaled` (n × 450)
- **QC grid**: plots 20 random downsampled ERSPs so you can sanity-check that the gross spectro-temporal pattern survived
- Calls `R.fit_and_save` twice (KMeans + HC, same K-sweep)
- Outputs land in `outputs/clustering/{kmeans,hierarchical}/rawds/runs/<id>/`
- **BACKFILL_CENTROIDS**: per-cluster mean of the 15×30 downsampled ERSP as the chip thumbnail

**Knobs in the config cell**:
- `FREQ_BAND_EDGES` — override the default band list with custom `(lo_hz, hi_hz)` tuples
- `FMAX_HZ` — upper edge of the ORIGINAL ERSP freq axis (default 500.0; set to 400 if your STFT capped there)
- `TIME_BINS_OUT` — change time-axis target length

**Expectation per Becht et al. 2019**: silhouettes 2–4× higher than the 210 full-res raw runs.

---

## 230 — `02_FBM_Clustering/230_blob_clustering.ipynb`

**Role**: cluster on **valley-segmented blob features** (positions + spreads of significant peaks/valleys in each ERSP).

- Loads canonical sample set
- Calls `s22_build_blob_feature_matrix` (in `lf_blob_metrics.py`):
  - Segments each ERSP into up to `max_blobs=6` peaks/valleys via valley-following
  - For each blob: 8 features (t_start, f_peak, t_peak, sf, st, cov, mean, area_norm)
  - Returns `X_blob` shape `(n, 6 × 8 = 48)` and a parallel `blobs_per_sample` list of blob dicts
- **No score gating** — all canonical samples enter (was a `s30_gate_by_blob_score` step before; removed for parity with the other notebooks)
- **Per-sample alternate-view PNGs** cell:
  - For each sample, writes `ERSP_blob/<cond>/<stem>_BLOB.png` (transparent ERSP + q34-style blob overlays — solid vertical/horizontal bars for spread, dashed full-height vertical lines at blob time-edges, red ● for positive, blue ▪ for negative)
  - And `ERSP_minus101/<cond>/<stem>_M101.png` (painted -1/0/+1 segmentation)
  - These power MOBA's samples-pane Blob and −101 view toggles
  - Idempotent (skip-if-exists by default)
- Applies feature-type weights via `s23_init_blob_metric` → `Xw = X_blob * BLOB_WEIGHT_VEC`
- Calls `R.fit_and_save` twice (KMeans + HC K-sweep), feature_set=`blob`
- Saves `blob_artifacts/blobs_per_sample.joblib` + `segmentation_config.json` into each run dir (231 can optionally reuse these, but doesn't have to)
- **BACKFILL_CENTROIDS**: for each cluster finds the **medoid** sample (closest in `Xw` space to cluster mean) and renders that sample's actual ERSP + blobs as the chip thumbnail

---

## 231 — `02_FBM_Clustering/231_minus101_clustering.ipynb`

**Role**: cluster on **painted −1 / 0 / +1 segmentation maps** (mask-based representation of the blobs).

- Loads canonical sample set
- Calls `s22_build_blob_feature_matrix` inline (same params as 230 → identical blobs; 231 does NOT depend on 230 having run first)
- Calls `s23_build_minus101_feature_matrix`:
  - For each ERSP, paints +1 in pixels covered by a positive blob, −1 in pixels covered by a negative blob, 0 elsewhere
  - Resizes to `SCALE` (default 1.0 = full ERSP resolution, 129×300; can set lower to downsample)
  - Returns `X_101` flattened
- QC: `q60_plot_minus101_overlays` shows 30 random samples (ERSP + painted overlay) for visual sanity
- Calls `R.fit_and_save` twice (KMeans + HC K-sweep), feature_set=`minus101`
- **BACKFILL_CENTROIDS**: per-cluster mean of the painted map (reshape `X_101.mean(axis=0)` to `ds_shape`)

**Note**: very similar information to `blob` (same valley segmentation). Useful for exploration but largely redundant with `blob` for paper-track work.

---

## 232 — `02_FBM_Clustering/232_hg_clustering.ipynb`

**Role**: cluster on **high-gamma (HG) band time series** — one 1D vector per electrode. Closest to the iEEG gold-standard analysis (Crone et al. 1998, Hamilton et al. 2018).

- Loads canonical sample set
- Calls `lf_hg.build_hg_feature_matrix`: for each ERSP, averages across freq bins inside `HG_BAND = (70, 150) Hz` → 1D vector of length `n_time` (300)
- Returns `X_hg` shape `(n, 300)`
- **Per-sample HG sparkline PNGs**: writes `ERSP_hg/<cond>/<stem>_HG.png` for each sample (powers MOBA's HG view toggle)
- Calls `R.fit_and_save` twice (KMeans + HC K-sweep), feature_set=`hg`
- **BACKFILL_CENTROIDS**: per-cluster mean HG sparkline (line plot with shaded zero-cross fill) as the chip thumbnail

**Knobs**:
- `HG_BAND` — change band range
- `FMAX` — upper edge of the original ERSP freq axis (default 500 Hz)

---

## 211 — `02_FBM_Clustering/211_validation.ipynb`

**Role**: **post-hoc validation** of every existing clustering run. Doesn't fit any new clustering — it adds diagnostic artifacts into each run dir that MOBA's Stats tab surfaces.

Runs three analyses per run, each in its own cell with a `SKIP_EXISTING` guard so it's safe to re-run:

### A. Consensus stability (Monti et al. 2003 + Hennig 2007)
- For each run: refits KMeans `N_RUNS=50` times with different random seeds at the run's best K
- Builds the `(n_samples, n_samples)` co-clustering matrix — fraction of runs in which each pair of samples ended up together
- Per-cluster Jaccard stability = mean intra-cluster co-occurrence
- Writes `consensus_matrix.npy`, `consensus_heatmap.png` (samples reordered by cluster — diagonal blocks = stable, off-diagonal smear = unstable), `per_cluster_stability.csv`, `stability_summary.json`
- **Interpretation**: Jaccard ≥ 0.7 = stable cluster, 0.5–0.7 = borderline, < 0.5 = artifact of one specific seed

### E. Tibshirani gap statistic (Tibshirani et al. 2001)
- For each run with a `k_range` in its manifest, computes the gap statistic across the whole K range
- Compares observed within-cluster dispersion `W_k` to expected `W_k*` on uniform-bounding-box null data (with `N_REFS=10` null replicates per K)
- Writes `gap_by_k.json` + `gap_by_k.png` (line with error bars, best K by Tibshirani rule marked)
- Default config skips raw + minus101 (slow at 38k features); runs on hg + blob only
- **Best K rule**: smallest K such that `Gap(K) ≥ Gap(K+1) − s_{K+1}`

### C. Anatomical purity (one-time aparc cache + per-cluster scoring)
- One-time precompute (first cell): `build_aparc_cache` projects every electrode in `fsaverage/coords/*.csv` to the nearest fsaverage cortical vertex, reads its Desikan-Killiany aparc label, writes `outputs/250_recon/fsaverage/aparc_lookup.csv`. Requires `mne` (auto-fetches fsaverage on first call, ~50 MB).
- For every run: joins `labels.csv` with the aparc cache, computes per cluster:
  - **Top 3 regions** + proportions
  - **Shannon entropy** (low = anatomically coherent)
  - **Purity** (proportion of the modal region — 0..1)
  - Optional **permutation p-value** vs anatomical-scattered null (`N_PERM=500`)
- Writes `per_cluster_anatomy.csv` + `per_cluster_anatomy.json` into each run dir

After running, push the new artifacts. MOBA's **Stats tab** then shows curves + per-cluster table with green/grey/red pills.

---

## 252 — `02_FBM_Clustering/252_clustering_recon.ipynb`

**Role**: 3D **brain reconstruction** per cluster. Renders the fsaverage cortex with each cluster's electrodes plotted as colored spheres.

- Walks `index.json` for every clustering run
- For each run: joins `labels.csv` rows to the cached fsaverage coords from `outputs/250_recon/fsaverage/coords/*_contacts_fsaverage.csv` via `(patient, contact_name)`
  - Contact-name normalization: both sides go through `normalize_label` (strip `_` and `-`, uppercase) so `aH_R-1` matches `AHR1`
  - Reports `UNMATCHED_contacts.csv` per run for QC
- For each cluster: renders four views of the fsaverage pial surface with electrodes overlaid:
  - `by_patient/<view>.png` — colored by patient
  - `by_condition/<view>.png` — colored by condition
  - Views: `lateral_L`, `lateral_R`, `dorsal`, `frontal`
- Writes `recon/` subfolder inside each run dir + a `recon_summary.json`
- Skips runs that already have populated `recon/` unless `SKIP_IF_RECON_EXISTS = False`
- Uses MNE-Python + PyVista for off-screen rendering. Requires `mne`, `pyvista`, `pyvistaqt`.

**Important**: this notebook **only consumes** pre-computed fsaverage coords; it does not generate them. Coords come from a one-time prep step (no longer in this folder — the old `250_recon_shared_data_WM.ipynb` was deleted because it referenced missing config attributes; coords were preserved).

---

## Run order

**On a fresh patient set:**
1. `140` (analysis) — extract ERSPs (slow: hours)
2. `210` — full-res raw clustering (this also builds the `_dataset/canonical/` cache that 212/230/231/232 reuse)
3. `212`, `230`, `231`, `232` — parallelizable in any order (all hit the canonical cache instantly)
4. `211` — post-hoc validation on every run
5. `252` — 3D brain recon

**After just tweaking a clustering algorithm or feature_set**: only re-run that one notebook + `211` + `252` if needed.

**After changing the canonical filter (e.g. high-activity thresholds)**: delete `outputs/_dataset/canonical/`, re-run any one clustering notebook to rebuild the cache, then re-run the others.

---

## Where the outputs go

| Artifact | Where | What it powers |
|---|---|---|
| Per-electrode ERSP `.npy` | `01_FBM_Analysis/outputs/04_ersp_LM_RAWONLY/<pid>/LM/ERSP_matrix/` | Input to every clustering notebook |
| Per-electrode display PNGs | Same tree, `ERSP_clean/`, `ERSP_blob/`, `ERSP_minus101/`, `ERSP_hg/` | MOBA samples-pane toggle (4 views) |
| Canonical-dataset cache | `02_FBM_Clustering/outputs/_dataset/canonical/` (gitignored) | Shared by 210/212/230/231/232 |
| Clustering runs | `02_FBM_Clustering/outputs/clustering/<method>/<feature_set>/runs/<id>/` | MOBA dropdown + Stats tab |
| Run index | `02_FBM_Clustering/outputs/clustering/index.json` | MOBA reads to populate dropdown |
| fsaverage coords + meshes | `02_FBM_Clustering/outputs/250_recon/fsaverage/` | 252 reads; MOBA 3D brain renders |
| Aparc lookup cache | `02_FBM_Clustering/outputs/250_recon/fsaverage/aparc_lookup.csv` | 211 anatomy purity |
| Validation artifacts (per run) | Run dir: `consensus_*`, `per_cluster_stability.csv`, `gap_by_k.*`, `per_cluster_anatomy.*` | MOBA Stats tab |

---

## What each function module does (helpers under `02_FBM_Clustering/functions/`)

| Module | Role |
|---|---|
| `lf_dataset.py` | `prepare_dataset()` — the canonical sample loader. Single source of truth. |
| `lf_features.py` | `downsample_ersp_to_bands()` + `FREQ_BANDS_15_TO_400HZ` — used by 212 |
| `lf_blob_metrics.py` | Valley segmentation (`s22_*`), per-sample blob render (`render_blob_overlay`, `save_sample_blob_png`), cluster-density render — used by 230 |
| `lf_minus101.py` | -1/0/+1 paint (`paint_minus101_map`), per-sample M101 PNG render — used by 230 + 231 |
| `lf_hg.py` | HG band extraction (`extract_hg_time_series`, `build_hg_feature_matrix`), sparkline render — used by 232 |
| `lf_stability.py` | Consensus matrix + per-cluster Jaccard (Monti + Hennig) — used by 211 |
| `lf_anatomy.py` | Aparc cache builder + per-cluster anatomy purity — used by 211 |
| `lf_cluster_run.py` | The orchestrator — `fit_and_save()` (KMeans + HC + K-sweep + manifest + index update + gap stat), `load_run()`, `assign_new()` — used by every clustering notebook |
| `lf_recon_shared.py` | 3D rendering helpers — used by 252 |
