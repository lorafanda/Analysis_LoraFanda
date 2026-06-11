# Classification pipeline — what each notebook does

This folder runs **supervised** experiments on the same per-electrode ERSPs that the
clustering stage (`02_FBM_Clustering`) uses. Where clustering asked *"what structure is
in the data?"*, classification asks two specific questions and reports how well the
classes separate.

> 📖 **How to read every plot/metric** (definitions, equations, what to look for):
> [`READING_GUIDE.md`](READING_GUIDE.md).

---

## Pipeline overview

```
01_FBM_Analysis/140 → ERSP_matrix/*.npy per electrode×condition
                          │
                          ▼
03_FBM_Classifying/310 → assemble + cache classification features
                          │           (reuses lf_dataset / lf_features / lf_hg from 02)
              ┌───────────┴────────────┐
              ▼                         ▼
            320                        330
   condition decoding         parcellation decoding
   (audio/picture/reading)    (Yeo-7 & Yeo-17)
              └───────────┬────────────┘
                          ▼
                  390 results (read me) · 391 compare features across tasks
```

Everything is decoded with **nested GroupKFold by patient** (outer = held-out test
patients, inner = hyper-parameter tuning), **two classifiers** (logistic regression +
random forest), and a **nested, time-matched family of eight feature variants** so both
"full spectrum vs high-gamma" *and* "pattern vs amplitude" are fair tests:

| variant | freq × time | representation | dims |
|---|---|---|---|
| `full_300` / `full_30` | 15 bands × {300, 30} | continuous | 4500 / 450 |
| `hg_300` / `hg_30` | 1 line (70–150 Hz) × {300, 30} | continuous | 300 / 30 |
| `full_300_rn` / `full_30_rn` | 15 bands × {300, 30} | **row-normalised** (unit-L2 per sample) | 4500 / 450 |
| `m101_300` / `m101_30` | 15 bands × {300, 30} | **discretized −1/0/+1** (score-gated blobs) | 4500 / 450 |

Two axes of comparison:
- **Frequency content** — matched pairs `full_300 vs hg_300`, `full_30 vs hg_30` (time held
  fixed); `full_300 vs full_30` isolates time resolution. ⚠️ Frequency and time are coupled
  by the STFT, so even matched grids aren't a perfectly clean separation (caveat in 390).
- **Amplitude confound** — the triad `full_* → full_*_rn → m101_*`. Column z-scoring can't
  remove a sample's overall magnitude, so a strong electrode decodes on *loudness* not
  *shape*. Row-norm removes magnitude (keeps graded shape); the discretized map removes
  amplitude entirely, painting **only score-gated significant blobs** as ±1 (low-score /
  noisy segments stay 0). A big drop continuous→discretized = the result was amplitude-driven.

---

## 310 — `310_classification_dataprep.ipynb`

**Role**: build the classification-ready feature matrices once and cache them.

- Loads every electrode × condition ERSP (ungated) for patients that have **all three**
  conditions, via `lf_classify.prepare_full_dataset` (wraps `lf_dataset.prepare_dataset`
  with `apply_high_activity=False`; the high-activity flag is still computed per sample).
- **Condition task** (gated): one sample per high-activity electrode × condition →
  caches `X/y/groups/meta/cols` per variant.
- **Parcellation task**: one sample per electrode = `audio ⊕ picture ⊕ reading`
  concatenated, for each variant × {Yeo-7, Yeo-17}. An electrode is kept iff it has all
  three conditions, is high-activity in ≥1 condition, and sits in a **real Yeo network**
  (medial-wall / white-matter / unknown dropped). Yeo labels are read straight from
  `02_FBM_Clustering/outputs/250_recon/fsaverage/coords/<pid>_contacts_fsaverage.csv`
  (`yeo7_network` / `yeo17_network`), joined on a normalized contact name (same
  `normalize_label` rule as 252).
- Cache lives in `outputs/_dataset/classification/` (gitignored, machine-local).

**Run this first whenever the upstream ERSPs or the high-activity/Yeo filters change.**

---

## 320 — `320_condition_classification.ipynb`

**Role**: decode the stimulus condition (audio / picture / reading) from one electrode's
response. 3 classes × 8 variants × 2 classifiers = **16 runs** (subset via `VARIANTS_TO_RUN`).

- Loads the cached condition arrays, calls `lf_classify.run_experiment('condition', …)`.
- Writes a run dir per (variant × classifier) with confusion matrix, per-class metrics,
  permutation null, feature importance (see "Outputs" below).

## 330 — `330_parcellation_classification.ipynb`

**Role**: decode the Yeo functional network of an electrode from its **3-condition
concatenated** profile. {Yeo-7, Yeo-17} × 8 variants × 2 classifiers = **32 runs** (subset via `VARIANTS_TO_RUN`).

- Classes are imbalanced → `class_weight='balanced'`, headline = balanced accuracy / macro-F1.
- `run_experiment('parcellation_yeo7' | 'parcellation_yeo17', …)`.

## 390 — `390_results.ipynb`

**Role**: the **results page**. Computes nothing heavy — pulls the artifacts 320/330 wrote
and lays them out with the narrative and a guide to reading each figure (confusion matrix,
per-class strength, permutation null, feature importance), plus an at-a-glance comparison
across all experiments. Re-run after 320/330 to refresh; it always loads the latest run per
(task · variant · classifier).

## 391 — `391_compare_features.ipynb`

**Role**: **compare the 8 feature variants across class types** (condition / Yeo-7 / Yeo-17).
Reads the runs 320/330 wrote (`compare_table` → `index.json` + each `metrics.json`); no new
compute. Three chance-normalized figures, saved to `outputs/classification/_compare/`:
- **Fig 1 heatmap** — variant × (task × classifier), cell = fraction above chance (+ BA, `*`).
- **Fig 2 forest** — one panel per task, balanced accuracy ± 95% CI, filled = p<.05.
- **Fig 3 paired contrasts** — amplitude triad (continuous→row-norm→discretized) + time (300 vs 30).

Every panel plots **fraction above chance** or one panel per task — chance differs by task,
so raw balanced accuracy is never shared across tasks on one axis.

---

## Outputs

Each run lands in
`outputs/classification/<task>/<variant>/<classifier>/runs/<timestamp>/`:

| Artifact | What it is |
|---|---|
| `manifest.json` | run config + CV scheme + summary metrics; also appended to `outputs/classification/index.json` |
| `metrics.json` | overall (balanced accuracy + bootstrap CI, macro-F1, accuracy, chance, permutation p) + per-class records |
| `confusion_matrix.{csv,png}` | row-normalized confusion matrix |
| `per_class_metrics.csv` + `per_class_strength.png` | per-class recall/precision/F1, one-vs-rest ROC AUC, bootstrap CI, FDR-corrected permutation p with significance stars |
| `permutation_null.{json,png}` | grouped label-permutation null for balanced accuracy + per class |
| `feature_importance.{csv,png}` (+ `_by_band/_by_condition/_by_time.csv`) | what drove separation: LR signed per-class coefficients, or RF impurity + permutation importance, aggregated by condition / band / time |
| `class_feature_heatmap.{csv,png}` | **heatmap** rows = classes (regions/conditions), cols = frequency band × condition, colour = per-class mean response |
| `class_ersp_profile.png` | per-class **full-spectrum ERSP**, conditions **concatenated** (`[audio｜picture｜reading]`); dashed grey line at 50% of each block = stim→response boundary (first half = sensing, second half = response). Always full-spectrum, even on HG runs |
| `coef_heatmap.{csv,png}` (LR) | **heatmap** of class × band signed linear weights — what the model uses |
| `predictions.csv` | out-of-fold `y_true / y_pred / fold` per sample |

---

## Validation & statistics (the "how strong is each class" part)

- **Nested GroupKFold by patient** — no patient in both train and test; scaler fit on
  train folds only; hyper-parameters tuned on an inner patient-grouped split.
- **Balanced accuracy** (mean per-class recall) is the headline, with a **bootstrap-over-
  patients** 95% CI.
- **Above-chance significance** — grouped label-permutation null, empirical p for overall
  balanced accuracy **and per class** (recall), the latter **FDR-corrected** (Benjamini-
  Hochberg) across classes.
- **Per-class separability** — recall ± CI, one-vs-rest ROC AUC, and the confusion-matrix
  row tell you which classes genuinely separate and which collapse into neighbours.

---

## Helper module — `functions/lf_classify.py`

| Piece | Role |
|---|---|
| `prepare_full_dataset` | load all electrode×condition ERSPs (patients with all 3 conditions), keep the per-sample high-activity flag |
| `build_condition_arrays` / `build_parcellation_arrays` | turn ERSPs into the two task matrices (+ feature-column metadata) |
| `ersp_to_feature` / `feature_columns` | the 8 feature variants (continuous / row-normed / discretized × 2 time grids + HG) and their column maps (reuses `lf_features` / `lf_hg` / `lf_blob_metrics` / `lf_minus101` from 02) |
| `resolve_m101_score_min` / `ersp_to_minus101` | score-gated −1/0/+1 discretization — paints only high-score blobs |
| `load_yeo_lookup` | per-electrode Yeo-7/17 labels from the coords CSVs |
| `make_estimator` | LR + RF pipelines (scaler folded in) and their hyper-parameter grids |
| `nested_cv_predict` / `compute_metrics` / `permutation_test` | the validation engine |
| `feature_importance` | LR coefficients / RF importances mapped back to condition·band·time |
| `run_experiment` | one experiment end-to-end → saved run dir + index update |
| `list_runs` / `load_run` | read artifacts back for the 390 narrative |
| `save_arrays` / `load_arrays` | the feature cache written by 310, read by 320/330 |

`_build_notebooks.py` regenerates the four notebooks from source (stdlib only); safe to
delete — it is not part of the runtime pipeline.
