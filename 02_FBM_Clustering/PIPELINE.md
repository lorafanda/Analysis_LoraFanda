# The pipeline, step by step

Every terminal run, in order, with what it writes, who reads it, and what has to exist
first. Paths are relative to the repo root. Run from the repo root; `python` is the Python
3.11 on the laptop, the LORA_ENV2 interpreter on the server.

**Two machines.** The server has the data and no site checkout or git remote. The laptop
has the site and the remote. Steps 1–14 run on either; 15–19 are laptop only.

**Steps 1–3 are the ones that fail silently.** Nothing downstream can tell a stale cube
from a current one, so a skipped audit shows up weeks later as a cohort that is quietly
wrong.

---

## A · Preprocessing — one patient at a time

### 1. `141_audit_140.py` — is any patient's run older than its config?

```bash
python 01_FBM_Analysis/141_audit_140.py
```

| | |
|---|---|
| **needs** | an `outputs/03_ERSP` tree |
| **writes** | `outputs/03_ERSP/audit_140.tsv` · `audit_140.md` |
| **read by** | `make_s1_tab.py`, `make_s1_patients.py`, `make_s1_procedure.py`, `make_signal_ersp_block.py`, `142_stale_cubes.py`, `scripts/make_lm_review_bundle.py` |

One row per patient: reference and its source, bad list, Unknown drops, notch, trials in /
kept / used, cubes and shapes. **Config is read live**, so the bad-list and reference
columns say what the code would do *now* — a row that disagrees with its own run means
that patient needs re-running. Since 2026-10-13 it reads the newest run that actually
FINISHED (`Pipeline done` plus an OK report line), not simply the newest log, so an
interrupted run no longer stands in for the real one.

### 2. `142_stale_cubes.py` — what a rerun left behind

```bash
python 01_FBM_Analysis/142_stale_cubes.py            # list
python 01_FBM_Analysis/142_stale_cubes.py --delete   # and delete
```

| | |
|---|---|
| **needs** | step 1 (it reads `audit_140.tsv` for each patient's run time) |
| **writes** | nothing without `--delete`; deletes files with it |
| **read by** | — (its output is the tree itself) |

140 never clears a patient folder. A channel a rerun no longer produces — newly
bad-listed, newly a reference contact, newly Unknown — keeps its cubes, halves, CLEAN
images and QC figures from the earlier run, and those enter the next cohort build as if
they were current. v8 carried 48 of them. `--patients` narrows it.

### 3. `140_ersp_pipeline.py` — the analysis, per patient

```bash
python 01_FBM_Analysis/140_ersp_pipeline.py --patient PAT_3455
```

| | |
|---|---|
| **needs** | raw data; `functions/config.py` current for that patient |
| **writes** | `outputs/03_ERSP/<pid>/LM/` — `ERSP_matrix/<cond>/*.npy` (the cubes), `ERSP_halves/`, `ERSP_clean/`, `ERSP/`, `PerTrial/{HFA,Signal,TrialScores,Report,Photodiode}/`, `PSD_raw/`, `PSD_clean/`; `outputs/03_ERSP/wm_reref_report.tsv`; `logs/<pid>_<stamp>.log` |
| **read by** | everything — `rebuild_concat_cache.py` walks `ERSP_matrix`, 141 reads the logs and the report, `01_FBM_LM_report.py` and the site read the figures |

Use the id **config** spells: MicroEPI patients are `G-01` … `G-06`, not `PAT_6854`.
Several at once with `--patient EL034 PAT_3455`, or one per terminal in parallel.
`--pd` re-extracts the photodiode triggers and rewrites prep0 · `--no-qc` cubes only ·
`--no-export` QC only · `--no-psd` skips the slow dpi-600 montages · `--macros-only`
skips the microwires.

**A reference contact gets no products.** If a bad-list change touches the white-matter
mean, every channel of that patient changes and a rerun is the only fix; if it doesn't,
deleting that contact's files is enough (step 4).

### 4. `143_delete_non_neural.py --bad-listed` — remove a contact without a rerun

```bash
python 01_FBM_Analysis/143_delete_non_neural.py --bad-listed --patients PAT_3455 --delete
```

| | |
|---|---|
| **needs** | a contact on `cfg.bad_channels_manual` that is **not** in the reference |
| **writes** | deletes that contact's cubes, halves, images, figures and trial-score rows; logs `outputs/03_ERSP/logs/bad_listed_deleted_<stamp>.tsv` |
| **read by** | `make_s1_patients.py` reads the deletion log — it is how the site's per-patient table knows what went and when |

Deleting by hand loses that record. Without `--bad-listed` it does the same for the
non-neural channels (EKG-, EMG-, photo, E1–E4) that 140's old aux filter let through.

### 5. `147_keep_all_channels.py` — the audit tree *(optional)*

```bash
python 01_FBM_Analysis/147_keep_all_channels.py --patients PAT_3455 --keep-wm --fast
```

| | |
|---|---|
| **needs** | the same inputs as 140 |
| **writes** | `outputs/03_ERSP-allchan-wm/<pid>/LM/…` |
| **read by** | `make_contact_review_block.py`, `148`, eyes |

140 with the bad list **cleared**. The only place a bad-listed or reference contact is
drawn, so it is where you look before deciding whether a contact deserves the list.
`--keep-wm` keeps the reference contacts as data · `--fast` writes Signal, HFA and trial
scores only.

### 6. `148_contact_noise_audit.py` — which contacts are noisy *(optional)*

```bash
python 01_FBM_Analysis/148_contact_noise_audit.py --all --report
```

| | |
|---|---|
| **needs** | `03_ERSP` (it re-references internally) |
| **writes** | `outputs/contact_noise/<pid>_contact_noise.tsv` |
| **read by** | `make_contact_review_block.py`, and by you |

Broadband and 70–150 Hz sd **after** subtracting the white-matter mean, as a multiple of
that patient's own montage median. The only view where a noisy contact shows: the cubes
are dB re baseline and divide the noise out, the raw signal is dominated by the shared
reference, and `score_hg` is a within-channel z so every contact reads 1.0.

---

## B · The cohort

### 7. `rebuild_concat_cache.py` — build the dataset the clustering reads

```bash
python 02_FBM_Clustering/rebuild_concat_cache.py --dry-run --version 16
python 02_FBM_Clustering/rebuild_concat_cache.py --apply  --version 16
```

| | |
|---|---|
| **needs** | steps 1–3 clean |
| **writes** | `outputs/_dataset/concat_source_v<N>/` (the cache) · `outputs/clustering/cohort_v<N>/cohort_v<N>_summary.json` · `cohort_v<N>_per_patient.csv` · `name_match_*.csv`; **and calls `make_cohort_figure.py`**, which writes FIG C.0 |
| **read by** | `lf_concat.DEFAULT_CONCAT_CACHE` → steps 8–10 and everything they feed |

**A new directory every time, never a rebuild in place.** `prepare_dataset` decides a
cache hit on parameters only — no patient list, no timestamps, no content hash — so an
in-place rebuild would print "cache hit" and hand back the old activity gating.

Nothing needs editing afterwards: `DEFAULT_CONCAT_CACHE` picks the highest
`concat_source_v<N>` on disk by itself, unless `LF_CONCAT_CACHE` is set in the
environment, which overrides it.

**After this every run on disk is stale**, because each was fitted on the previous cohort.

---

## C · The fits — three terminals, independent

The four standard feature sets differ in **frequency resolution only**, on one shared
30-bin time grid:

| set | bands | features |
|---|---|---|
| `concat_bands15` | 15 | 1350 |
| `concat_bands5` | 5 | 450 |
| `concat_bands2` | 30–50 + 70–150 Hz | 180 |
| `concat_hfa` | 70–150 Hz | 90 |

### 8. `240_cluster_kmeans.py` / 9. `241_cluster_hierarchical.py` / 10. `242_cluster_cnmf.py`

```bash
python 02_FBM_Clustering/240_cluster_kmeans.py
python 02_FBM_Clustering/241_cluster_hierarchical.py
python 02_FBM_Clustering/242_cluster_cnmf.py
```

| | |
|---|---|
| **needs** | step 7. **242 also needs 240 first** — `run_decomposition.py` takes its matrix from the newest k-means run of that feature set |
| **writes** | `outputs/clustering/<method>/<feature_set>/runs/<stamp>/` — `cluster_labels_by_k.csv`, `labels.csv`, `X_train.npy`, `metrics.json`, `manifest.json`, `sweep_metrics.csv`, silhouettes and centroid PNGs; cNMF adds `loadings_by_k/G_k<NN>.npy`, `components_by_k/`, `G_loadings.npy`, `components.npy` |
| **read by** | steps 11–19, and `lf_runs.newest_run` is how every one of them finds the current run |

K = 5…30 over all four sets. `--ks 8` for one K · `--feature-set concat_hfa` for one set ·
`--no-heldout` to skip the long part. A set that already has a run on the current cache is
skipped.

**Each one calls, in order:**

| sub-step | writes | read by |
|---|---|---|
| `run_decomposition.py` *(242 only — this IS the cNMF fit)* | `outputs/clustering/decomposition/<fset>/` — `cv_rank_curve.csv`, `lopo_patients.csv`, `lopo_null.csv`, `component_correlation.csv`, F3a/F3b/F4/F5 PNGs, `index.json` | `choose_k.py`, `make_decomposition_figure.py`, `make_cnmf_method_block.py`, `make_overview_options.py` |
| `publish_decomposition.py` *(242 only)* | turns that into a registered run under `cnmf/<fset>/runs/<stamp>/` | everything that resolves runs |
| `sweep_decomposition.py` *(242 only)* | `loadings_by_k/`, `components_by_k/`, `cluster_labels_by_k.csv`, `sweep_by_k.csv` | FIG 1/2/4/5, statistics, bundles |
| `make_heldout_variance.py` *(all three)* | `outputs/clustering/bsf_comparison/heldout_variance_<method>.csv`, `heldout_variance_summary_<method>.csv`, `heldout_peaks_<method>.csv` | step 12, `make_heldout_figure.py`, `make_cluster_webblock.py`, `make_graphical_abstract.py` |

### 10b. The HFA time-bin sweep *(optional, cNMF only)*

```bash
python 02_FBM_Clustering/242_cluster_cnmf.py --feature-set concat_hfa300 concat_hfa200 concat_hfa150 concat_hfa100 concat_hfa30 concat_hfa10
```

The same 70–150 Hz band at another time resolution — the time axis is an anti-aliased
resize, so any bin count works. Not in the default set list: a plain run never fits them.
`concat_hfa30` is identical to `concat_hfa`; drop it and reuse that fit if you like.

**240 has to run on them first**, same as for any other set — `run_decomposition.py` takes
its matrix from the newest k-means run of that feature set, so 242 alone fails with
`no runs directory for kmeans/concat_hfa300`:

```bash
python 02_FBM_Clustering/240_cluster_kmeans.py --feature-set concat_hfa300 concat_hfa200 concat_hfa150 concat_hfa100 concat_hfa30 concat_hfa10
```

---

## D · Choosing K

### 11. `choose_k.py` — three criteria side by side *(optional)*

```bash
python 02_FBM_Clustering/choose_k.py
```

| | |
|---|---|
| **needs** | step 7 (it refits from the cache) |
| **writes** | `outputs/clustering/bsf_comparison/` — `k_bicv.csv`, `k_stability.csv`, `k_split.csv`, `K_selection.png` |
| **read by** | the site's clustering tab; you |

Bi-cross-validation, leave-one-patient-out stability, split-half replication. They answer
different questions, so agreement between them is worth more than any one.

### 12. `249_cluster_statistics.ipynb` — the K rule and the full battery

Jupyter, Run All. **Run only after 240, 241 and 242 have all finished.**

| | |
|---|---|
| **needs** | steps 8–10; its cell 3 reads the three `heldout_variance_<method>.csv` |
| **writes** | `bsf_comparison/heldout_variance_ALL.csv` · `peak_k.json` · `peak_k_heldout.json` · `patient_dominance_by_k.csv`; then per set `outputs/clustering/statistics/<fset>_K<k>/` |
| **read by** | `heldout_variance_ALL.csv` → `00_Paper2_Figures.py`, `make_heldout_figure.py`, `make_k_tradeoff_figure.py`, `make_cluster_figures.py`. `peak_k.json` → `make_heldout_figure.py`, `make_paper_tab.py`, `make_paper_figures_webblock.py`, `make_s2_gallery.py` |

**Cell 3 picks K by patient dominance, not by held-out variance** (since 2026-10-13): the
largest K at which no cluster is more than half one patient, read off convex NMF and used
for all three methods. The held-out peak is still computed and printed for comparison —
it never turns over, so it only ever nominates the top of the sweep.

### 13. `make_cluster_statistics.py` — one set at a time

```bash
python 02_FBM_Clustering/make_cluster_statistics.py --feature-set concat_bands15 --k 7 --n-null 50 --lopo-reps 10
```

| | |
|---|---|
| **needs** | steps 8–10, and the fsaverage coordinate table (step 18) for the coherence number |
| **writes** | `outputs/clustering/statistics/<fset>_K<k>/` — `stats_separation.csv`, `stats_agreement.csv`, `stats_coherence.csv`, **`stats_lopo_predict.csv`**, `stats_lopo.csv`, `stats_cluster_sizes.csv`, `stats_labels.npy`, `stats_summary.json`, plus one JSON per run for the visualizer |
| **read by** | `make_cluster_figures.py`, `make_cluster_webblock.py`, the cluster visualizer |

249 calls this per set; run it directly to redo one. **`stats_lopo_predict.csv` is the
generalization test**: fit without a patient, place that patient's electrodes with the
other patients' components, and ask whether they land where the full fit put them. The
older refit test (`stats_lopo.csv`) never touches the held-out patient, so a cluster that
exists only because a patient is in the fit passes it. Its null now permutes whole
**shafts**, not single electrodes.

Use each set's clean K from step 14 — at a K where a cluster is one patient, LOPO is
partly re-measuring the dominance you already measured.

---

## E · Figures

### 14. `make_k_tradeoff_figure.py` — the two criteria, every set on one page

```bash
python 02_FBM_Clustering/make_k_tradeoff_figure.py
python 02_FBM_Clustering/make_k_tradeoff_figure.py --feature-set concat_hfa300 concat_hfa200 concat_hfa150 concat_hfa100 concat_hfa30 concat_hfa10
```

| | |
|---|---|
| **needs** | `heldout_variance_ALL.csv` (step 12 cell 3) and the runs' `loadings_by_k` |
| **writes** | `outputs/clustering/paper_figures/FIG_K_tradeoff_<method>.png` + `_caption.txt` |
| **read by** | you — it is not on the site yet |

Left: held-out variance over K, one line per set, marker at each set's own peak. Right:
red = clusters over 50% one patient, orange = electrodes in them, line style per set,
triangle at the largest K with none. Reads only; safe to run while 249 is going.
`--danger` moves the 50% · `--method` switches algorithm.

### 15. The paper figures

```bash
python 02_FBM_Clustering/00_Paper2_Figures.py
python 02_FBM_Clustering/00_paper2_figures2_2.py --k 7
python 02_FBM_Clustering/00_paper2_figure3_lana.py
python 02_FBM_Clustering/00_paper2_figure4_correspondence.py --k 7
python 02_FBM_Clustering/00_paper2_figure4_panels.py --k 7
python 02_FBM_Clustering/00_paper2_figure5_confusion.py --k 5 6 7 8 9 10
```

| | |
|---|---|
| **needs** | steps 8–10; FIG 1 needs `heldout_peaks_cnmf.csv`; FIG 2/3 and FIG 1 need the coordinate table (step 18) for their brain panels |
| **writes** | `outputs/clustering/paper_figures/` — `FIG4_correspondence_K<NN>_run<id>.png`, `FIG4_matched_*.csv`, `FIG4_corr_*.csv`, `FIG4_summary_*.csv`, `FIG4_K<NN>.png`, `FIG5_pairs_*.csv`, `FIG5_KbyK_*.csv` and the figure PNGs |
| **read by** | `make_paper_figures_webblock.py`, `make_paper_tab.py` → the site |

FIG 4 is split on purpose: `correspondence` does the permutations and writes tables,
`panels` redraws from them in seconds. Every K writes its own `K<NN>` files, so the
correspondence runs are safe in parallel — **but FIG 5 must be ONE process** with all the
K values, because it also writes `FIG5_pairs_allK` and `FIG5_KbyK` aggregated over
whatever it was given.

**FIG 0, FIG 1, FIG 2 and FIG 3 render brains** (PyVista/VTK) and need the screen awake —
they segfault on a locked screen. Don't run them next to 252.

### 16. `make_paper2_outputs.py` — all of the above in one command

```bash
python 02_FBM_Clustering/make_paper2_outputs.py --list
python 02_FBM_Clustering/make_paper2_outputs.py --headless     # server
python 02_FBM_Clustering/make_paper2_outputs.py --only site    # laptop
```

43 steps in order, each its own process, with the exit code and the time for every one.
Phases: `audit` → `select` → `runs` → `paper` → `stage02` → `norm` → `prep` → `bundles` →
`site`. `--headless` skips the five that render brains. The machine split is automatic:
the `site` phase runs only where `~/lorafanda.github.io` exists.

---

## F · Recon and the brain views

### 17. `250_recon_fsaverage.ipynb` — electrode coordinates on fsaverage

Jupyter, Run All.

| | |
|---|---|
| **needs** | FreeSurfer recons on the share |
| **writes** | `outputs/250_recon/fsaverage/coords/<pid>_contacts_fsaverage.csv`, **`ALL_PATIENTS_contacts_fsaverage.csv`**, `ALL_PATIENTS_contacts_fsaverage_nowm.csv`, `_regen_qc.tsv` |
| **read by** | `make_cluster_statistics.py` (coherence), `make_coverage_bundle.py`, `00_Paper2_Figures.py`, `make_patient_roster.py`, `make_decomposition_figure.py`, `make_bsf_comparison.py`, `functions/lf_counters.py`, 252 |

**`ONLY` in cell 1 replaces the table, it does not add to it.** Step 2 aggregates with
`pd.concat` over the patients *it just projected*, so running with `ONLY = ["EL051",
"EL052"]` leaves a two-patient `ALL_PATIENTS` file and everything downstream silently
matches ~60 electrodes. Set `ONLY = []` for a full rebuild.

### 18. `252_clustering_recon.ipynb` — the 3-D brain per cluster

Jupyter, with `RUN_FILTER = [{'feature_set': '<fset>', 'method': 'cnmf'}]`.

| | |
|---|---|
| **needs** | steps 8–10 and 17 |
| **writes** | `<run>/recon/cluster_<NN>/`, `UNMATCHED_contacts.csv`, `recon_summary.json`, `*__with_fsaverage.csv` |
| **read by** | MOBA's 3-D brain, the cluster visualizer |

Reads the **no-WM** coordinate table, so an electrode that is in the cohort but
anatomically white matter is reported as unmatched and does not appear. Needs the screen
awake.

---

## G · The site — laptop only

### 19. Bundles, then the page

```bash
python 02_FBM_Clustering/check_site_ready.py --k 8
python 02_FBM_Clustering/audit_site_figures.py
python 02_FBM_Clustering/make_coverage_bundle.py
python 02_FBM_Clustering/make_centroid_bundle.py
python 02_FBM_Clustering/scripts/push_cluster_bundle.py --commit --push
python 02_FBM_Clustering/make_cohort_figure.py --insert
python 02_FBM_Clustering/make_paper_figures_webblock.py --insert
python 01_FBM_Analysis/make_s1_tab.py --insert
python 02_FBM_Clustering/make_site_ui.py --insert
```

| | |
|---|---|
| **needs** | everything above; `make_s1_tab` needs step 1 |
| **writes** | `outputs/250_recon/fsaverage/coverage_viz/` and `outputs/clustering/paper_web/centroids/`; the `--insert` scripts splice blocks into `~/lorafanda.github.io/analysis_status.html` |

`check_site_ready` marks every item OK / STALE / MISSING; `audit_site_figures` names the
script that regenerates each stale figure, so its output is the task list.

**`make_site_ui.py --insert` goes last** — it is the page's runtime layer and wraps
whatever blocks are on the page at the time.

**A track missing from `TRACKS` in `make_coverage_bundle.py` is invisible to the
visualizer** however complete its runs are. New feature sets need adding there.

The page fetches every figure from
`raw.githubusercontent.com/lorafanda/Analysis_LoraFanda/main`, so an uncommitted figure is
a broken image. Commit with an explicit pathspec — the index holds thousands of files that
are not yours to commit:

```bash
git commit --only -F <message file> -- <paths>
```

---

## The short order

```
141 → 142 --delete → 140 (per patient)          A
rebuild_concat_cache --dry-run → --apply         B
240 → 242, and 241                               C   (242 needs 240 first)
249  →  make_cluster_statistics per set          D
make_k_tradeoff_figure, the paper figures        E
250 → 252                                        F
bundles → push → the --insert scripts → site     G
```

Skipping a step does not fail loudly. That is the whole reason this file exists.
