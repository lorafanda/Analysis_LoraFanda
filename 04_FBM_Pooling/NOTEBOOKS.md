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
              490   → results narrative (boxcar vs gaussian robustness)
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
- For each condition, overlays **every contact's blobs** as **red(+) / blue(−) ellipse
  outlines** in the freq × time plane (shade ∝ |mean dB|), plus a **time-marginal** density
  (contacts active per bin, split by sign). The dashed line at bin 150 = response onset.
- Final cell: **you edit** the three zones — `perception`, `pre_articulation`, `audio` —
  each with a `boxcar` (`t_lo_pct`/`t_hi_pct`) and a `gaussian` (`center_pct`/`sigma_pct`),
  all in **% of the 0–300 axis**. Saved to `outputs/pooling/window_config.json`.

**Run this first; everything downstream reads the window config it writes.**

---

## 420 — `420_pool_and_qualify.ipynb`

**Role**: pool power inside each window and qualify each contact.

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
| `default_window_config` / `save_window_config` / `load_window_config` / `validate_window_config` | the window-config JSON written by 410, read by 420 |
| `segment_contact_blobs` / `blob_ellipse` / `plot_blob_overlay` / `plot_time_marginal` | the 410 overlays (reuses `lf_blob_metrics.s21_segment_valley_blobs`) |
| `make_window_weights` / `window_support` | boxcar / gaussian time-weight vectors + the gate support mask |
| `feature_vector` / `pooled_power` | the two feature sets (`hg` via `lf_hg`, `bands15` via `lf_features`) + time-weighted pooling |
| `windowed_gate` | clustering's σ/proportion gate, restricted to the window |
| `temporal_null_p` | per-contact circular-time-shift null p (optional) |
| `build_pool_table` / `qualifier_summary` / `load_pool_table` | the 420 driver + summary + cache I/O |
| `ensure_aparc_cache` / `load_coords` / `attach_anatomy` | the anatomy joins (Yeo from coords, DK from `lf_anatomy.build_aparc_cache`) |
| `zone_to_cluster_id` / `summarize_anatomy` | per-zone purity + compactness (reuses `lf_anatomy.save_*_artifacts`) |
| `render_zone_brains` | self-contained fsaverage PyVista render of qualifiers |
| `new_run_dir` / `update_index` / `list_runs` / `latest_run` | run-dir + index plumbing (mirrors `lf_classify`) |

`_build_notebooks.py` regenerates the four notebooks from source (stdlib only); safe to
delete — it is not part of the runtime pipeline.
