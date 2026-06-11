# Pooling pipeline — what each notebook does

This folder is the **fourth, confirmatory** stage of the FBM pipeline. Where clustering
(`02`) asked *"what structure is in the ERSPs?"* and classification (`03`) asked *"can we
decode condition / network?"*, pooling asks a **hypothesis-driven** question:

> Do contacts show a task-related response in **a-priori time zones** of the warped trial
> timeline, and do the responsive contacts **cluster anatomically**?

It runs on the **same canonical samples** the clustering pipeline uses — one sample = the
trial-averaged ERSP per electrode × condition, shape **129 freq × 300 time**, time-warped
**50% stimulus / 50% response** (bin 150 = response onset). ERSP values are already
baseline-relative, so significance is a **window-restricted** version of clustering's
high-activity gate.

---

## Pipeline overview

```
01_FBM_Analysis/140  → ERSP_matrix/*.npy per electrode × condition
                          │  (same files clustering + classification use)
                          ▼
04_FBM_Pooling/410   → blob overlays per condition → HAND-DEFINE windows
                          │            (outputs/pooling/window_config.json)
                          ▼
              420   → pool power in each window (boxcar + gaussian),
                       qualify each contact (windowed clustering gate),
                       two feature sets (hg, bands15)
                          ▼
              430   → map qualifiers to Yeo-7/17 + Desikan-Killiany gyri,
                       purity / compactness, fsaverage renders
                          ▼
              440   → region × time cascade raster (thesis figure)
                          ▼
              490   → results narrative (boxcar vs gaussian robustness)

   (parallel path)
              450   → predefined time-frequency ROI pooling
                       (cluster/condition-blind; 11 physiology ROIs, both grids)
                          ▼
              460   → concatenated [audio|picture|reading] functional-role matching
                       (condition-structured; strict conjunction on score-gated −1/0/+1)
                          ▼
              web/pool.html  → POOL brain page (Niivue, colour-by-role) for lorafanda.github.io
```

Two axes of comparison run throughout:
- **Window shape** — `boxcar` (primary, equal weight) vs `gaussian` (robustness; centre-
  weighted, tolerant of edge effects / latency jitter). Agreement = the zone is real.
- **Feature set** — `hg` (70–150 Hz line) vs `bands15` (the 15 bands separately).

---

## 410 — `410_zone_discovery.ipynb`

**Role**: *see* the data and hand-pick the pooling windows.

- Loads the canonical, **ungated** dataset (`prepare_pooling_dataset`, wraps
  `lf_dataset.prepare_dataset` with `apply_high_activity=False`).
- **Resolution toggle `USE_DS`** (top of the notebook): `True` (default) works on the fast
  **15×30 band-downsampled** grid (`downsample_dataset`); `False` uses the full-res 129×300.
  - **full-res:** overlays **every contact's blobs** as **red(+) / blue(−) ellipse outlines**
    (shade ∝ |mean dB|), thinned by a tunable **`SCORE_PCT`** blob-score gate
    (`resolve_score_gate`), plus a **time-marginal** (contacts active per bin, split by sign).
  - **ds:** per-condition **mean band×time heatmap** (`plot_ds_heatmap`) + a mean ±power
    **time-marginal** (`plot_ds_time_marginal`), since blobs don't survive downsampling.
- Final cell: **you edit** the three zones — `perception`, `pre_articulation`, `audio` —
  each with a `boxcar` (`t_lo_pct`/`t_hi_pct`) and a `gaussian` (`center_pct`/`sigma_pct`),
  all in **% of the axis**. Saved to `outputs/pooling/window_config.json`.

**Run this first; everything downstream reads the window config it writes.**

---

## 420 — `420_pool_and_qualify.ipynb`

**Role**: pool power inside each window and qualify each contact.

- Honours the same **`USE_DS`** toggle (`grid='ds'|'full'`); ds caches to
  `pool_table_ds.parquet`, full to `pool_table_full.parquet`. ⚠️ On `ds` the σ-gate runs on the
  smoothed band-mean map (a coarse proxy) — use `USE_DS=False` for final qualification numbers.
- **Pooling** = a **time-weighted average** of the contact's power over the window. Two
  feature sets, run identically: `hg` (1 line) and `bands15` (15 bands, native time axis).
- **Qualification** = clustering's gate (`prop(>2.2σ) ≥ 0.02` **OR** `prop(<−3.0σ) ≥ 0.04`)
  computed **only over the window's columns**. Both signs kept; gaussian gate support = ±2σ.
- Writes a tidy table (one row per contact × condition × zone × shape × feature) to
  `outputs/_dataset/pooling/pool_table.parquet` + a `pool` run dir with `qualifier_summary`.
- Optional `N_PERM > 0` adds a per-contact **circular-time-shift null p** (secondary).

> ⚠️ *Comparable, not identical* to clustering: the windowed gate yields **lower** qualifier
> counts than whole-ERSP high-activity — that restriction is the confirmatory test.

---

## 430 — `430_anatomy_mapping.ipynb`

**Role**: map qualifiers to anatomy and summarize where they cluster.

- Joins qualifiers (on `patient_id` + normalized contact name) to **Yeo-7 & Yeo-17**
  (precomputed in the coords CSVs), **Desikan-Killiany gyri** (`ensure_aparc_cache`, built
  once via MNE + fsaverage), and fsaverage **xyz**.
- Per-zone **purity / entropy** (`lf_anatomy.cluster_anatomy_purity`) and **spatial
  compactness** (`cluster_spatial_compactness`), once per window shape so boxcar vs gaussian
  are directly comparable.
- **fsaverage surface renders** of qualifying contacts per condition × zone, red(+)/blue(−).

> ⚙️ Requires `mne` + `pyvista` + the fsaverage surfaces (server-side). 410/420/490 don't.

---

## 440 — `440_cascade.ipynb`

**Role**: the confirmatory result as a **publication figure** — a region × warped-time **cascade
raster**.

- For each anatomical region (`region_labels`: Yeo-7 default, or Yeo-17 / Desikan-Killiany),
  averages the **high-gamma time-course** across its **responsive** contacts (`responsive_contacts`
  from the `420` pool table), one matrix per condition (`build_cascade`).
- Rows are **sorted by peak latency**, so the perception → pre-articulation → audio cascade reads
  top → bottom; the a-priori zones are shaded behind (`plot_cascade`). A companion table
  (`cascade_table`) gives per-region contact counts + peak latencies.
- **No spatial interpolation** — every row is a real average over real contacts (the honest,
  review-proof alternative to a smoothed surface video). Saved under
  `outputs/pooling/cascade/<scheme>/runs/<id>/`. Honours the same `USE_DS` toggle.

---

## 450 — `450_roi_pooling.ipynb` (parallel path)

**Role**: the **cluster-blind, condition-blind** pooling — a fixed, physiology-driven library of
time-frequency **ROIs** applied to every electrode × condition ERSP.

- ROIs live in `functions/roi_config.py` (`ERSP_POOLING_PARAMS`, both `ds` and `full` grids): each
  a 2-D box (`f_rows × t_bins`, **1-based inclusive**) + a **sign** hypothesis (`pos`/`neg`/`both`),
  citation-anchored to a documented marker.
- §1 draws the **ROI map** (`plot_roi_map`) — labelled `(a)(b)(c)…` boxes coloured by sign, + a
  `roi_legend` table, + `validate_roi_config` (flags ds↔full Hz / sign mismatches).
- §2 `build_roi_table` pools (`pool_roi` = box-mean dB) + sign-gates (`roi_gate`) every sample.
- §3 **option A**: a contact *expresses* an ROI if it qualifies in ≥1 condition (`roi_counts`).
- §4 `roi_region_crosstab` → ROI × Yeo-7 (no MNE) "where each signature lives."
- **Additive** — does not touch the zone-based `410–490` path. ⚠️ Reconcile the two flagged
  predeterminant issues (alpha_beta + broadband ds↔full mismatch; broadband box-mean ≠ Manning index).

---

## 460 — `460_role_matching.ipynb` (condition-structured)

**Role**: assign each contact a **functional role** from its `[audio|picture|reading]` pattern.

- Concatenates the three conditions per contact (only contacts with **all three**), **score-gated
  discretizes** to −1/0/+1 (`discretize_ersp`, reuses `lf_minus101` + the `SCORE_PCT` blob gate),
  and assigns a role by **strict conjunction** (`build_role_table`): every box of a role's template
  (`functions/roi_config_concatenated.py`) must clear the **proportion gate** in its sign.
- Roles (draft): **auditory** (audio-stim + all 3 responses), **visual** (picture+reading stim),
  **motor** (all 3 responses), **multimodal** (all 3 stim). `role` = most-specific match.
- §4 `export_pool_web` writes `contacts_pool.csv` + `pool_index.json` (xyz + role + colour) — the
  data the **POOL** page reads. Reuses the MOBA fsaverage meshes.

### `web/pool.html` — the POOL brain page
Niivue page (sibling of MOBA's `moba.html`) that renders contacts on fsaverage **coloured by role**
with per-role filters. Drop it into the `lorafanda.github.io` repo; set the three URLs in its
`CONFIG` block to match how MOBA serves meshes + per-run CSVs.

---

## 490 — `490_results.ipynb`

**Role**: the **results page**. Pulls the artifacts 410/420/430 wrote, lays them out with the
narrative and a guide to reading each figure, and shows the **boxcar-vs-gaussian** robustness
comparison (qualifier counts + anatomy purity/compactness side by side). Re-run after 420/430.

---

## Outputs

```
outputs/pooling/  index.json · window_config.json
├── discovery/runs/<id>/   overlay_<cond>.png · time_marginal_<cond>.png
├── pool/runs/<id>/        pool_table.parquet · qualifier_summary.csv · manifest.json
└── anatomy/<yeo7|yeo17>/<shape>/runs/<id>/
      per_cluster_anatomy.{csv,json} · per_cluster_spatial_compactness.{csv,json}
      cluster_id_map.json
   anatomy/renders/runs/<id>/renders/<cond>/<zone>_<shape>/<view>.png

outputs/_dataset/pooling/   pool_table.parquet + the ungated ERSP cache   (gitignored)
outputs/_anatomy/aparc_lookup.csv   one-time MNE build                    (gitignored)
```

---

## Helper module — `functions/lf_pool.py`

| Piece | Role |
|---|---|
| `prepare_pooling_dataset` | load the canonical (df_meta, X_3d), ungated, + a `contact_norm` join key (reuses `lf_dataset.prepare_dataset`) |
| `downsample_dataset` | band-downsample the full ERSPs to the 15×`DS_TIME_BINS` `ds` grid (reuses `lf_features.build_X_3d_downsampled`) |
| `default_window_config` / `save_window_config` / `load_window_config` / `validate_window_config` | the window-config JSON written by 410, read by 420 |
| `segment_contact_blobs` / `resolve_score_gate` / `blob_ellipse` / `plot_blob_overlay` / `plot_time_marginal` | full-res 410 overlays + the tunable blob-score gate (reuses `lf_blob_metrics.s21_segment_valley_blobs`) |
| `plot_ds_heatmap` / `plot_ds_time_marginal` | the `USE_DS` 410 views (mean band×time heatmap + power time-marginal) |
| `make_window_weights` / `window_support` | boxcar / gaussian time-weight vectors + the gate support mask |
| `feature_vector` / `pooled_power` | the two feature sets (`hg` via `lf_hg`, `bands15` via `lf_features`) + time-weighted pooling |
| `windowed_gate` | clustering's σ/proportion gate, restricted to the window |
| `temporal_null_p` | per-contact circular-time-shift null p (optional) |
| `build_pool_table` / `qualifier_summary` / `load_pool_table` | the 420 driver + summary + cache I/O |
| `ensure_aparc_cache` / `load_coords` / `attach_anatomy` | the anatomy joins (Yeo from coords, DK from `lf_anatomy.build_aparc_cache`) |
| `zone_to_cluster_id` / `summarize_anatomy` | per-zone purity + compactness (reuses `lf_anatomy.save_*_artifacts`) |
| `render_zone_brains` | self-contained fsaverage PyVista render of qualifiers |
| `region_labels` / `responsive_contacts` | per-contact region (Yeo/aparc) + the responsive-contact set (440) |
| `build_cascade` / `cascade_table` / `plot_cascade` | the 440 region × time cascade raster + its companion table |
| `ROI_PARAMS` / `validate_roi_config` | the predefined ROI library (`roi_config.py`) + ds↔full consistency check (450) |
| `pool_roi` / `roi_gate` / `build_roi_table` | 2-D box-mean pooling + sign-directional gate, condition/cluster-blind (450) |
| `roi_contact_summary` / `roi_counts` / `roi_region_crosstab` | option-A expression + per-ROI counts + ROI × region (450) |
| `plot_roi_map` / `roi_legend` | the labelled ROI-box figure + its legend (450) |
| `ROLE_PARAMS` / `discretize_ersp` / `build_concat_discretized` | concatenated-role config + score-gated −1/0/+1 + per-contact concat (460) |
| `box_expresses` / `role_matches` / `build_role_table` / `role_counts` | strict-conjunction matching + role assignment (460) |
| `export_pool_web` / `role_colors` | POOL web data export (`contacts_pool.csv` + `pool_index.json`) feeding `web/pool.html` |
| `new_run_dir` / `update_index` / `list_runs` / `latest_run` | run-dir + index plumbing (mirrors `lf_classify`) |

`_build_notebooks.py` regenerates the four notebooks from source (stdlib only); safe to
delete — it is not part of the runtime pipeline.
