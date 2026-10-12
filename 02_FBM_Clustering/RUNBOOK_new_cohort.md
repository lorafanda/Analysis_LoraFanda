# Runbook — a new cohort, script by script

Run from the repo root. `python` means the Python 3.11 you use on that machine
(`C:/Users/fanda/AppData/Local/Programs/Python/Python311/python.exe` on the laptop, the
LORA_ENV2 interpreter on the server).

---

## The preprocessing, per patient

```bash
python 01_FBM_Analysis/140_ersp_pipeline.py --patient PAT_3455
```

Runs the whole analysis for that patient: raw file → trials → bad-channel drop →
reference → notch → ERSP cubes, CLEAN images, HFA and Signal rasters, trial scores, PSD
and montage QC. Takes several patients at once (`--patient EL034 PAT_3455`) and the id is
the one **config** spells — MicroEPI patients are `G-01` … `G-06`, not `PAT_6854`.
Other options: `--pd` re-extracts the photodiode triggers first and rewrites prep0,
`--no-qc` writes cubes only, `--no-export` writes QC only, `--no-psd` keeps the QC but
skips the slow dpi-600 PSD montages, `--macros-only` skips the microwires on MicroEPI.

```bash
python 01_FBM_Analysis/141_audit_140.py
```

The audit of the 140 runs: one row per patient saying what it was run with and what came
out — reference and its source, bad list, Unknown drops, notch, trials in / kept / used,
cubes and shapes. Config is read **live**, so the bad-list and reference columns say what
the code would do *now*: a patient whose row disagrees with its own run needs re-running.
Writes `outputs/03_ERSP/audit_140.tsv` and `.md`. No options. Run it before the site
rebuild — `make_s1_tab.py` reads that TSV.

```bash
python 01_FBM_Analysis/142_stale_cubes.py
```

Looks for stale files: anything older than its patient's own 140 run. 140 never clears a
folder, so a channel a rerun no longer produces — newly bad-listed, newly a reference
contact, newly Unknown — keeps its cubes, halves, CLEAN images and QC figures from the
earlier run, and those enter the next cohort build as if they were current (v8 carried
48 of them). Add `--delete` to delete them, `--patients EL034 PAT_3455` to narrow it.

```bash
python 01_FBM_Analysis/143_delete_non_neural.py --bad-listed --patients PAT_3455
```

Removes one contact's products from the tree **without** a rerun: cubes, halves, images,
QC figures and its rows in the trial-score tables. Use it for a contact bad-listed after
its patient was run, and only when nothing else depends on it — not a reference contact,
because the notch and the mean are still decided with it. Without `--bad-listed` it does
the same for the non-neural channels (EKG-, EMG-, photo, E1–E4) that 140's old aux filter
let through. Lists by default; add `--delete`. It writes
`outputs/03_ERSP/logs/bad_listed_deleted_<stamp>.tsv`, which is where the site's
per-patient table learns what went and when — so use this rather than deleting by hand.

```bash
python 01_FBM_Analysis/144_move_to_pertrial.py --apply
```

Moves `HFA/`, `Signal/`, `TrialScores/` and `Report/` under `<pid>/LM/PerTrial/`, which is
where 140 has written them since 2026-10-02. Only needed for a patient last run before
that. Lists by default; `--patients`, `--tree <name>` for another tree. Merges if the
destination exists and the newer file wins.

```bash
python 01_FBM_Analysis/147_keep_all_channels.py --patients PAT_3455 --keep-wm --fast
```

140 with the bad list **cleared**, into `outputs/03_ERSP-allchan-wm`. This is the audit
tree: it is the only place a bad-listed or reference contact gets drawn, so it is where
you look at a contact to decide whether it deserves to be on the list. `--keep-wm` also
keeps the white-matter reference contacts as data channels, `--bad-in-ref` puts the bad
ones back into the mean, `--fast` writes only Signal, HFA and trial scores, `--tree` names
another output tree.

```bash
python 01_FBM_Analysis/148_contact_noise_audit.py --all --report
```

Measures each contact's broadband and 70–150 Hz sd **after** subtracting the white-matter
mean, over the task window, as a multiple of that patient's own montage median. That is
the only view where a noisy contact shows: the cubes are dB re baseline and divide the
noise out, the raw signal is dominated by the shared reference, and `score_hg` is a
within-channel z so every contact reads 1.0. Writes
`outputs/contact_noise/<pid>_contact_noise.tsv`. `--patients` for a few, `--thresh` sets
the cut `--report` prints at.

```bash
python 01_FBM_Analysis/make_contact_review_block.py --insert
python 01_FBM_Analysis/make_contact_review_block.py --wm-ref PAT_3975 --insert
```

Puts the contacts worth judging on the status page, full width, with the clean ERSP, the
HFA raster and the Signal traces side by side and the 148 numbers beside them. The first
form shows each patient's worst `--top 8` over `--thresh`; the second shows one patient's
**whole** white-matter reference. The figures are copied into
`outputs/_contact_review/` and must be committed, or the page shows nothing.

```bash
python 01_FBM_Analysis/01_FBM_LM_report.py
```

The per-patient PDF: every contact's clean ERSP, HFA raster and Signal traces, four to a
page, bad channels marked red and reference contacts amber, with the photodiode check on
page 2. `--tree` for another tree.

---

## The cohort

```bash
python 02_FBM_Clustering/rebuild_concat_cache.py --dry-run --version 16
python 02_FBM_Clustering/rebuild_concat_cache.py --apply --version 16
```

Builds the dataset the clustering reads, as a **new** directory
(`outputs/_dataset/concat_source_v16`), and re-computes the activity gate from the cubes
on disk. It must be a new directory: `prepare_dataset` decides a cache hit on parameters
only — no patient list, no timestamps, no content hash — so a rebuild in place would print
"cache hit" and hand back the old gating. The dry run names the patients entering and
leaving and the ones excluded by `lf_concat.DEFAULT_EXCLUDE_PATIENTS`; the apply prints
the gated cohort, the `X_concat` shape and a per-patient table with the deltas against
v15. **It calls `make_cohort_figure.py`** itself at the end, so FIG C.0 is never a version
behind. Nothing needs editing afterwards: `lf_concat.DEFAULT_CONCAT_CACHE` picks the
highest `concat_source_v<N>` on disk by itself, unless `LF_CONCAT_CACHE` is set in the
environment, which overrides it.

---

## The clustering

```bash
python 02_FBM_Clustering/240_cluster_kmeans.py
python 02_FBM_Clustering/241_cluster_hierarchical.py
python 02_FBM_Clustering/242_cluster_cnmf.py
```

The three algorithms, one script each, independent — three terminals. Each fits K = 5…30
over all four feature sets (`concat_hg`, `concat_rawds`, `concat_bands5`,
`concat_bands5z`), writes a run folder per fit, and then **calls
`make_heldout_variance.py`** for the bi-cross-validated held-out curve (`--tag <method>`,
so three parallel runs never write to the same CSV). Options: `--ks 8` for one K,
`--feature-set concat_bands5z` for one feature set, `--no-heldout` to skip the curve.
242 skips a feature set that already has a run (`SKIP_IF_RUN_EXISTS` in the file).

```bash
python 02_FBM_Clustering/choose_k.py
```

Three criteria for K side by side, because the one in use cannot work on its own: the
held-out peak puts 15–45 % of electrodes in one-patient clusters. K = 8 is the clean
choice. `--feature-set`, `--ks`, `--quick`, `--skip`.

`249_cluster_statistics.ipynb`, `250_recon_fsaverage.ipynb` and
`252_clustering_recon.ipynb` are still notebooks. 249 is the one that makes the report
cards, chips and rasters; run it in Jupyter when you want those.

---

## Everything else, in one command

```bash
python 02_FBM_Clustering/make_paper2_outputs.py --list
python 02_FBM_Clustering/make_paper2_outputs.py --headless      # on the server
python 02_FBM_Clustering/make_paper2_outputs.py --only site     # on the laptop
```

This is the orchestrator, and it is why the rest of this file is short: it runs the 43
steps below in order, each as its own process, and prints the exit code and the time for
every one. `--list` prints the plan and stops. `--headless` skips the five steps that
render brains (they need the screen awake — VTK segfaults on a locked screen).
`--only <phase>` runs one or more phases. `--ks 7 8 9` sets the K values FIG 4 and FIG 5
are drawn at. The two-machine split is automatic: the `site` phase runs only where
`~/lorafanda.github.io` exists, and `--site` / `--no-site` / `LF_SITE_DIR` override that.

What it calls, by phase:

| phase | scripts |
|---|---|
| `audit` | `141_audit_140.py` |
| `select` | `make_heldout_variance.py --spaces unit`, `choose_k.py` |
| `runs` | `make_missing_centroids.py --force`, `make_centroid_rasters.py` |
| `paper` | `00_paper2_figure0_coverage.py`, `00_Paper2_Figures.py`, `00_paper2_figures2_2.py` (×2), `00_paper2_figure3_lana.py`, `00_paper2_figure4_correspondence.py` and `00_paper2_figure4_panels.py` once per K, `00_paper2_figure5_confusion.py` |
| `stage02` | `make_separation_figure.py`, `make_separation_across_features.py`, `make_decomposition_figure.py`, `make_overview_options.py`, `make_component_anatomy.py`, `make_membership_explainer.py` |
| `norm` | `make_normalisation_figures.py` |
| `prep` | `make_preprocessing_figures.py` |
| `bundles` | `scripts/make_lm_review_bundle.py`, `make_coverage_bundle.py`, `make_centroid_bundle.py` |
| `site` | `make_cohort_figure.py --insert`, `make_patient_roster.py --insert`, `make_cluster_visualizer.py`, `make_paper_figures_webblock.py --insert`, `make_cluster_webblock.py --insert`, `make_s2_gallery.py --insert`, `make_paper_tab.py --insert`, `make_norm_notes.py --insert`, `make_s1_tab.py --insert`, `make_site_ui.py --insert`, `audit_site_figures.py --csv` |

`make_site_ui.py --insert` is last on purpose: it is the page's runtime layer (navigation,
sub-tabs, lightbox, phone layout) and it wraps whatever blocks are on the page at the
time, so it has to run after every `--insert`.

Two scripts worth running on their own:

```bash
python 02_FBM_Clustering/check_site_ready.py --k 8
python 02_FBM_Clustering/audit_site_figures.py
```

`check_site_ready` walks the whole chain and marks every item OK / STALE / MISSING —
STALE meaning the file is there but came from a superseded run or an older cohort.
`audit_site_figures` goes further and names the script that regenerates each figure, so
its output is the task list when something is out of date. Both read only.

---

## Pushing

```bash
python 02_FBM_Clustering/scripts/push_cluster_bundle.py
python 02_FBM_Clustering/scripts/push_cluster_bundle.py --commit --push
```

The visualizer fetches its data from `raw.githubusercontent.com/.../main`, so the
coverage and centroid bundles have to be on main before the page can show a new cohort.
This commits them in chunks the repo's guard accepts (22 MB per commit, 5 MB per file),
each one a `git commit --only` with an explicit pathspec — the index holds thousands of
files that are not yours to commit, so never a bare `git commit`. Prints the plan without
`--commit`. `--label` sets the cohort name in the messages.

Then the page itself, from the site checkout:

```bash
cd ~/lorafanda.github.io
git commit --only -F <message file> -- analysis_status.html
git push
```

---

## The order, short

1. `141_audit_140.py` — is any patient's run older than its config?
2. `142_stale_cubes.py --delete` — what a rerun left behind
3. `140_ersp_pipeline.py --patient <config id>` — for every patient 141 flagged
4. `rebuild_concat_cache.py --dry-run --version N`, then `--apply`
5. `240` / `241` / `242`, then `choose_k.py`
6. `make_paper2_outputs.py --headless` on the server
7. `make_paper2_outputs.py --only site` on the laptop
8. `scripts/push_cluster_bundle.py --commit --push`, then commit and push the page

Skipping a step does not fail loudly. That is the whole reason this file exists.
