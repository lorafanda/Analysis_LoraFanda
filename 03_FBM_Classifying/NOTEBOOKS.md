# Classification pipeline — what each notebook does

This folder runs **supervised** experiments on the same per-electrode ERSPs that the
clustering stage (`02_FBM_Clustering`) uses. Where clustering asked *"what structure is
in the data?"*, classification asks two specific questions and reports how well the
classes separate.

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
                        390  → results narrative (read me)
```

Everything is decoded with **nested GroupKFold by patient** (outer = held-out test
patients, inner = hyper-parameter tuning), **two classifiers** (logistic regression +
random forest), and **three feature variants**:

| variant | what | dims |
|---|---|---|
| `rawds` | band-aware downsampled ERSP (15 bands × 30 time) | 450 |
| `hg` | high-gamma 70–150 Hz band-mean time series | 300 |
| `hg_ds` | the HG series downsampled to 30 time bins | 30 |

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
response. 3 classes × 3 variants × 2 classifiers = **6 runs**.

- Loads the cached condition arrays, calls `lf_classify.run_experiment('condition', …)`.
- Writes a run dir per (variant × classifier) with confusion matrix, per-class metrics,
  permutation null, feature importance (see "Outputs" below).

## 330 — `330_parcellation_classification.ipynb`

**Role**: decode the Yeo functional network of an electrode from its **3-condition
concatenated** profile. {Yeo-7, Yeo-17} × 3 variants × 2 classifiers = **12 runs**.

- Classes are imbalanced → `class_weight='balanced'`, headline = balanced accuracy / macro-F1.
- `run_experiment('parcellation_yeo7' | 'parcellation_yeo17', …)`.

## 390 — `390_results.ipynb`

**Role**: the **results page**. Computes nothing heavy — pulls the artifacts 320/330 wrote
and lays them out with the narrative and a guide to reading each figure (confusion matrix,
per-class strength, permutation null, feature importance), plus an at-a-glance comparison
across all experiments. Re-run after 320/330 to refresh; it always loads the latest run per
(task · variant · classifier).

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
| `discriminative_maps.png` (rawds) | per-class **band × time** signature panels, in the same layout as the ERSPs |
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
| `ersp_to_feature` / `feature_columns` | the 3 feature variants and their interpretable column maps (reuses `lf_features` + `lf_hg` from 02) |
| `load_yeo_lookup` | per-electrode Yeo-7/17 labels from the coords CSVs |
| `make_estimator` | LR + RF pipelines (scaler folded in) and their hyper-parameter grids |
| `nested_cv_predict` / `compute_metrics` / `permutation_test` | the validation engine |
| `feature_importance` | LR coefficients / RF importances mapped back to condition·band·time |
| `run_experiment` | one experiment end-to-end → saved run dir + index update |
| `list_runs` / `load_run` | read artifacts back for the 390 narrative |
| `save_arrays` / `load_arrays` | the feature cache written by 310, read by 320/330 |

`_build_notebooks.py` regenerates the four notebooks from source (stdlib only); safe to
delete — it is not part of the runtime pipeline.
