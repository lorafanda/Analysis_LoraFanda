# Cleanup and pipeline plan, 2026-09-29 (first pass, nothing moved yet)

Inventory: 179 tracked .py / .ipynb under 01_FBM_Analysis (40) and 02_FBM_Clustering (139); no untracked
scripts; gitignored notebooks only in 02_FBM_Clustering/x_archive/. Every file below was classified from
its docstring, its last commit, who references it and whether the driver (make_paper2_outputs.py), the
site generators or the checker (check_site_ready.py) run it.

Confidence: **H** = safe to act on without looking again; **M** = act after a one-line check named in the
row; **L** = your call. "Archive" = move to x_archive/ with a one-line README entry; git keeps the history.

---

## A · The current chain (keep, 62 files)

### Stage 01, preprocessing
| file | role |
|---|---|
| 140_ersp_pipeline.py, 140_ERSP_analysis_pipeline.ipynb | the run (script = notebook; the notebook duplicates process_patient in two cells) |
| 141_audit_140.py, 142_stale_cubes.py | audit, orphan sweep |
| functions/config.py, lf_ersp.py, lf_io_utils.py, lf_trials.py, lf_micromacro.py, LF_pd.py, LFfunctions_PDextract.py | what 140 imports |
| el038_rebuild_triggers.py, el043_build_picture_triggers.py, el051_build_triggers.py, el052_build_triggers.py, pat_1327_build_triggers.py | trigger provenance per patient; config points at them. Keep, but in a `triggers/` folder |
| adjust_fixation_cross_duration.py, repair_trial_end_from_log.py | one-off repairs of trial_end (EL033, PAT_3965, G-02, G-03); provenance, same `triggers/` folder |
| make_s1_tab.py, make_preprocessing_figures.py, make_microepi_tab.py | site |

### Stage 02, clustering and site
| group | files |
|---|---|
| cache | rebuild_concat_cache.py, make_cohort_figure.py, check_recon_name_match.py, make_patient_meta.py, make_gate_examples.py |
| fits | 240_cluster_kmeans.ipynb, 241_cluster_hierarchical.ipynb, 242_cluster_cnmf.ipynb, sweep_decomposition.py, publish_decomposition.py, run_decomposition.py |
| K | make_heldout_variance.py, choose_k.py, sweep_stability.py, measure_cluster_stability.py |
| statistics (249) | 249_cluster_statistics.ipynb, make_cluster_statistics.py, make_cluster_figures.py, make_heldout_figure.py, make_missing_centroids.py, make_centroid_rasters.py |
| paper figures | 00_Paper2_Figures.py, 00_paper2_figure0_coverage.py, 00_paper2_figures2_2.py, 00_paper2_figure3_lana.py, 00_paper2_figure4_correspondence.py, 00_paper2_figure4_panels.py, 00_paper2_figure5_confusion.py, make_graphical_abstract.py |
| stage-02 figures (driver) | make_separation_figure.py, make_separation_across_features.py, make_decomposition_figure.py, make_overview_options.py, make_component_anatomy.py, make_membership_explainer.py, make_normalisation_figures.py, make_bands_figure.py |
| recon | 250_recon_fsaverage.ipynb, 211_validation.ipynb (aparc cache), 252_clustering_recon.ipynb, scripts/build_fsaverage_meshes.py |
| bundles | make_coverage_bundle.py, make_centroid_bundle.py, make_cluster_visualizer.py, make_lm_visualizer.py, scripts/precompute_activity_cube.py, precompute_contact_ersp.py, make_lm_review_bundle.py, make_lm_qc_images.py, push_review_bundle.py, serve_bundle.py |
| site | make_site_ui.py, make_site_gate.py, make_outline_tab.py, make_paper_tab.py, make_paper_figures_webblock.py, make_cluster_webblock.py, make_bsf_webblock.py, make_s2_gallery.py, make_norm_notes.py, make_kiss_tab.py, make_cnmf_method_block.py, make_cnmf_on_data_explainer.py, make_cnmf_three_roles_explainer.py, make_cnmf_explainer.py, make_hard_clustering_explainers.py (E1–E3 for the KISS tab), audit_site_figures.py, check_site_ready.py, make_paper2_outputs.py |
| functions | lf_dataset, lf_concat, lf_features, lf_hg, lf_decompose, lf_cluster_run, lf_runs, lf_centroids, lf_stability, lf_anatomy, lf_clustering (loaders), lf_recon_shared + _config (glass brains) |

---

## B · Archive (x_archive/, 31 files) — one-offs whose result is in git or on the site

| file | why | conf |
|---|---|---|
| compare_140_trees.py, attribute_140_changes.py, compare_140_hg.py, make_compare_140_figures.py | the 09-18 vs 08-16 tree comparison; report in outputs/compare_140/report.md | H |
| convert_tifs_to_png.py, move_halves_out_of_ersp_matrix.py | migrations done 08-26 / 09-07 | H |
| make_stale_cube_figure.py, make_late_band_figure.py, make_notch_explainer.py | one-off illustrations (stale cubes 08-25, PAT_6684's late band, why a stripe survives) | H |
| pd_detect_deriv_core.py | older copy of the detector; LFfunctions_PDextract carries `_pd_detect_deriv_core` | H |
| scripts/lookup_mni_to_fsaverage.py | the Lookup-MNI route, retired 09-29 (config's LOOKUP set is empty) | H |
| rename_hg_hfa.py, retire_sections.py, site_delete_blocks.py | site surgery done (09-05, 09-03, 09-17) | H |
| make_papers_deck.py, make_review_deck.py | slide decks of 08-24 | H |
| make_xcorr_figures.py, 215_concat_crosscorrelation.ipynb, 214_concat_timingranking.ipynb, functions/lf_cluster_timing.py | cross-correlation / timing track of August; **04_FBM_Pooling/make_role_onsets.py imports lf_cluster_timing** — archive the notebooks, keep the module until stage 04 is looked at | M |
| make_reliability_gate.py, make_reliability_examples.py, make_explainer_correlation.py | the reliability-gate alternative, decided against (noise-scaled gate won) | H |
| make_gate_split_figures.py, make_archetype_explainer.py, run_archetypes.py, functions/lf_archetypes.py | dropped from the driver 09-28; archetypes track not in the paper | M (say if archetypes stay a candidate) |
| make_explainer_halves.py, make_explainer_stability.py, make_explainer_three.py, make_correspondence_explainer.py | KISS explainers E4–E7, E11: images committed, generators unreferenced | M (only if the KISS tab keeps them as images) |
| make_pipeline_flowchart.py, make_rebuild_block.py, audit_run_artifacts.py | superseded by make_paper2_outputs.py --list and check_site_ready.py | M |
| make_cluster_cards.py, make_decomposition_drivers.py, make_recon_csv.py | MOBA-era helpers (211 / 237 only) | M |
| compare_band_schemes.py | the 15-vs-5 band decision, taken (concat_bands5) | H |
| scripts/precompute_region_overlays.py | Destrieux overlays for the activity page, unreferenced | L (do you still use the overlays?) |
| make_spin_sprite.py, pack_spin_frames.py | the rotating-sprite loading animation; render_cnmf_glassbrains.py stays in A (make_decomposition_figure's B3 panel embeds its renders, B2 reads 252's) | M |

---

## C · Delete candidates (git keeps them; 39 files)

| file | why | conf |
|---|---|---|
| 210, 212, 230, 231, 232, 233, 234, 235, 236, 237, 238 notebooks | pre-concat clustering tracks (raw, downsampled, blob, −1/0/+1, HG, evidence, decomposition drafts, ungated, over-factor); every result they produced was superseded by 240–242 + run_decomposition | H for 230 / 231 / 233 / 234 / 236 / 238; M for 210 / 212 / 232 / 235 / 237 (235 is referenced by publish_decomposition's docstring only) |
| 213_cluster_ranking.ipynb | iterative reassignment, abandoned | M |
| 251_recon_shared_data.ipynb | superseded by 250 since 09-04 (its own header says so) | H |
| 000_Paper2_figures.ipynb | the figure notebook the 00_*.py scripts replaced; the web block only mentions it in text | M (the [M] uncommitted edits in it are yours — look before deleting) |
| functions/lf_blob_clustering.py, lf_blob_clustering_config.py, lf_blob_metrics.py, lf_blob_recon.py, lf_blob_recon_config.py, lf_clustering_methods.py, lf_minus101.py | the blob and −1/0/+1 feature tracks; lf_blob_recon_config still points at the old Analysis_Lora path. Stages 03 / 04 / 05 mention these names but no file there imports them (checked with an import-line search, twice); lf_features.py mentions lf_minus101 in a docstring only | H |
| 02_FBM_Clustering/functions/lf_io_utils.py | the May copy of 01's loader; no 02 script imports it (checked: only three comments mention the name) | H |
| 01_FBM_Analysis: 11_MicroMacroComparison_Analysis.ipynb, 13_MicroSpikeSorting_Full.ipynb, 13_MicroSpikeSorting_NoTrials.ipynb, functions/lf_sort.py, lf_notch.py, lf_micro_io.py, lf_trials_qc.py, lf_recon.py | the MicroEPI spike-sorting side project; not part of the language-mapping paper | L — yours; if it is another paper, move the whole set to its own folder rather than delete |

Nothing in A is touched by B or C. Untracked outputs are not in this list.

---

## D · Dependencies outside 01 / 02 (why some deletions wait)

A word search finds lf_blob_clustering_config, lf_minus101, lf_hg, lf_cluster_timing and lf_recon_shared
mentioned in 03_FBM_Classifying (lf_classify), 04_FBM_Pooling (lf_pool, lf_atlas_corr, roi_config,
make_role_onsets) and 05_FBM_ResponseTiming (lf_rt). Two import-line searches (ripgrep and a full grep of
those folders) find no live import of any of them: the only cross-stage imports are 07_FBM_MotorM's
`from functions import lf_ersp / lf_io_utils` (01's). The mentions are path strings and comments. So the
deletions in C do not break 03 / 04 / 05 / 07; what those stages themselves become is a separate decision
(archive as a block, or keep 04 for the pooling paper).

---

## E · The clustering chain, made foolproof (proposal, nothing built yet)

What goes wrong today, seen this week: the fits are notebooks with hand-set cells (v12's driver ran on v11
fits and only FIG 4 refused); the aggregate cell of 250 rewrites the coordinate table from the run's
patients; 140 never clears a folder; the site phase is laptop-only; nine generators hard-code the site path.

One entry point, `02_FBM_Clustering/run_clustering.py`, subcommands in dependency order, every step
recording the cache version it consumed and refusing an input from another version (the FIG 4 rule
everywhere):

| step | command | what it wraps |
|---|---|---|
| 1 | `run_clustering.py cache --version 13` | 141 · 142 --delete (after listing) · rebuild_concat_cache · check_recon_name_match · make_cohort_figure |
| 2 | `run_clustering.py fit` | 240 / 241 / 242 as scripts: lf_cluster_run.fit_and_save for k-means, Ward, cNMF × 4 feature sets, K 5–30, unit-norm; run folders stamped with the cache version |
| 3 | `run_clustering.py select` | make_heldout_variance · choose_k; prints the peaks and stops until `--ks` is given |
| 4 | `run_clustering.py stats --ks 8` | 249 as a script: make_cluster_statistics · make_cluster_figures · make_heldout_figure |
| 5 | `run_clustering.py recon` | 252 as a script (needs the screen; keeps the notebook until VTK is headless) |
| 6 | `run_clustering.py figures --ks 8` | the driver's runs / paper / stage02 / norm / prep phases |
| 7 | `run_clustering.py bundles` | coverage → centroid → visualizer, LM review bundle, QC copies |
| 8 | `run_clustering.py site` | laptop only: the --insert scripts, make_site_ui last, then push_review_bundle and the main commits in size-capped chunks |
| — | `run_clustering.py status` | check_site_ready generalised: what is missing or older than the cache |

Rules that make it foolproof: (a) one cohort tag (`concat_source_v13`) written into every run folder,
figure caption and bundle manifest, checked on read; (b) no step deletes anything except `cache`'s 142
sweep, which lists first; (c) the site path from one place (LF_SITE_DIR or home) for all generators;
(d) the notebooks 240 / 241 / 242 / 249 become thin "run this cell" wrappers around the same functions, so
nothing is fitted twice or differently.

Effort: the fit and stats scripts are extraction from the notebooks (the functions already exist in
lf_cluster_run, lf_decompose, make_cluster_statistics); the driver already has phases 3, 6, 7, 8. About a
day of work, done in one branch, verified by reproducing the v12 fits bit-for-bit before v13.

---

## F · Questions to settle (in order)

1. Stages 03 / 04 / 05 / 07: archive as a block, or keep 04 (pooling) live? Decides half of section C.
2. MicroEPI spike sorting (11 / 13 and their five modules): its own folder, or archive?
3. Archetypes and the reliability gate: dead for the paper? (then B rows become deletions)
4. KISS explainers E4–E7, E11: keep the images only, or keep the generators?
5. Section E: go ahead on a branch, notebooks turned into scripts, before v13's fits — or after the paper?
