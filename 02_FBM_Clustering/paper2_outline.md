# Read me first
> How to read this tab: every line is a bullet; its darkness is how sure it is. Black = read from the code or from a run on 2026-09-24. Dark grey = the pipeline as I know it (not re-read today). Mid grey = plausible, check before you say it. Light grey = your slot: a citation with its section / paragraph, a number to read off a run, a decision. Edit `02_FBM_Clustering/paper2_outline.md`, then `python make_outline_tab.py --insert` and `python make_site_ui.py --insert`.
> Citation slots are written as chips, e.g. [cite: Leske & Dalal 2019, NeuroImage 189:763, sec 2.3 par 2]. Fill the section / paragraph, or replace the reference.
-4 Journal: Cell Reports (research article: Summary, Highlights, Introduction, Results, Discussion, Limitations, STAR Methods, Figures). Swap the order if the target changes.
-! Green = new since 2026-10-07 (a green bullet, or a green phrase inside an older one). Cohort **v15** exists (built 2026-10-09 13:25 from the 03_ERSP tree, the only tree since the 04_* trees were archived on 10-07): 1613 gated / 2976 ungated electrodes, 29 patients. Between v11 and v15: PAT_3780 re-timed (its 2025 trial tables were 180 s early) and rerun with five bad contacts, gated 49 → 92; EL031 processed and excluded; PAT_3415 contributes 0 (its depth contacts left the 03_ERSP run as Unknown); v13 (09-29) was the archived tree, v14 (10-07) still carried PAT_3780's wrong windows. All twelve clustering runs are refitted on v15 (2026-10-09); the held-out curves, 249, 252 and every figure are older. Stage-02 lines below say which cohort they are on.
-+ Cohort **v16** exists (built 2026-10-12 18:17, after the channel decisions and the three reruns): 1607 gated / 2946 ungated electrodes, 29 patients (PAT_3975 108 / 140 on its deep-WM reference, PAT_3455 73 / 86, PAT_6854 125 / 163). The script-by-script methods below (purple) are written against it; the twelve fits, the bundles and the figures are still to be redone on it.
-! 2026-10-12, channel decisions: 67 contacts bad-listed across 10 patients from the contact-noise audit (148) and the Signal / HFA / ERSP figures; PAT_3975 moved to a deep white-matter reference (11 contacts); PAT_6854, PAT_3975 and PAT_3455 rerun the same day, the other patients' removed contacts taken out with 143 (log `logs/bad_listed_deleted_20261012_163706.tsv`). v15 is therefore stale; **v16** is the next cache, and every stage-02 number waits for it.
- Cohort v11 existed (built 2026-09-26 21:33): 1487 gated / 3121 ungated electrodes, 30 patients. PAT_1327 joined, and the whole tree was re-preprocessed on the 2026-09-26 settings: post-offset floor 1 s → 0.2 s, spectrum interpolation per shaft for every patient, EL043 / EL046 bad contacts and re-scored trials.

# Summary
-3 Clinical hook: language must be mapped per patient before resection; electrical stimulation mapping is causal but slow, binary and incomplete [cite: Author et al., sec _ par _].
-3 Passive mapping from iEEG scores each electrode active / silent; the question here is what *kinds* of response exist.
-2 Data: 3 naming conditions (auditory definition, picture, written sentence completion), 30 patients, sEEG (+ 1 ECoG-only patient excluded), Bern + Geneva + Geneva micro–macro cohorts.
-2 Representation: one time-normalised ERSP cube per electrode × condition (103 freq × 300 warped bins, dB re baseline), concatenated audio | picture | reading.
-2 Method: convex NMF (graded), k-means and Ward on four feature sets; K by patient generalisation, not by held-out fit alone.
-3 Finding (current wording on the site): a small number of recurring response types (production-locked, auditory-onset, ramp-to-cue, suppressed, visual …); most survive feature set and algorithm changes; the weakest do not.
-4 Anatomy claim: none of the types is placed by the language atlas beyond a spatial null — decide whether this stays in the summary.
-4 Closing line candidate: "a response type is what an electrode does, not where it sits".

# Highlights
-4 Intracranial language responses cluster into a few recurring types (≤ 85 chars).
-4 Held-out fit nominates a K at which clusters are single patients.
-4 Types recur across feature sets and algorithms; the weak ones do not.
-4 No type sits in language-atlas cortex beyond a spatial null.

# Introduction
## 1 · Epilepsy surgery and functional mapping
-4 Drug-resistant focal epilepsy; resection; the boundary problem [cite: Author et al., sec _ par _].
-4 Why language cortex specifically: individual variability, atypical lateralisation, reorganisation [cite: Author et al., sec _ par _].
## 2 · Electrical stimulation mapping
-4 What ESM is; its strengths (causal, validated) [cite: Author et al., sec _ par _].
-4 Where it falls short: time, afterdischarges, coverage, binary read-out, scoring variability [cite: Author et al., sec _ par _].
## 3 · Passive mapping
-3 High-frequency activity (70–150 Hz) as a proxy for local population firing [cite: Ray & Maunsell 2011; Manning et al. 2009; Miller et al. 2009 — sec _ par _].
-4 Prior HFA-vs-ESM comparisons and their sensitivity / specificity [cite: Author et al., sec _ par _].
-3 The gap: existing passive maps report where, not what kind of response.
## 4 · This paper
-2 Question: do language responses fall into recurring types across patients, and which analysis choices they survive.
-2 What paper 1 established (the paradigm, the HFA map) — one paragraph, cite it.
-4 [cite: Fanda et al. (paper 1), sec _ par _]

# Results
## R1 · Task and cohort (FIG 0)
- Three conditions per trial block: auditory naming (spoken definition, German), picture naming, reading completion (written sentence). Each trial: stimulus → response cue ("?") → spoken answer; a fixation screen precedes the stimulus.
- Trial counts by design: 53 audio, 54 picture, 53 reading (51 picture for the Bern EL list); ~45 min per patient.
- 32 patients processed (5 MicroEPI G-01…G-06 without G-05, 16 Bern EL030–EL052, 11 Geneva PAT); cohort after rules: v9 = 28 patients / 2966 ungated / 1695 gated; v10 = 29 / 2961 / 1624; v11 = 30 patients / 3121 ungated / 1487 gated (PAT_1327 in at 101/165; PAT_3415's 24 depth contacts in, its 64 grid contacts out){{; v15 = 29 patients / 2976 ungated / 1613 gated (2026-10-09, the 03_ERSP tree: PAT_3780 re-timed, EL031 out, PAT_3415 at 0 because its 09-30 run dropped the depth contacts as Unknown)}}.
-! v15 per patient (gated / ungated): EL030 72/83, EL033 58/63, EL034 8/52, EL035 33/87, EL036 28/48, EL037 31/83, EL038 49/61, EL040 35/120, EL042 60/103, EL043 61/64, EL045 51/70, EL046 47/74, EL048 35/59, EL051 33/70, EL052 28/84, PAT_1327 111/164, PAT_2868 13/39, PAT_3066 37/152, PAT_3390 53/95, PAT_3455 73/93, PAT_3780 92/112, PAT_3965 77/169, PAT_3975 97/141, PAT_5515 9/143, PAT_5533 52/164, PAT_6619 58/133, PAT_6704 43/128, PAT_6854 137/175, PAT_6953 132/147.
- v11 per patient (gated / ungated): EL030 65/83, EL033 70/74, EL034 8/52, EL035 35/98, EL036 27/48, EL037 33/90, EL038 50/66, EL040 35/122, EL042 59/103, EL043 61/67, EL045 90/91, EL046 49/78, EL048 16/59, EL051 26/85, EL052 37/104, PAT_1327 101/165, PAT_2868 12/39, PAT_3066 34/152, PAT_3390 24/96, PAT_3415 24/24, PAT_3455 45/95, PAT_3780 48/116, PAT_3965 50/169, PAT_3975 76/155, PAT_5515 9/143, PAT_5533 49/164, PAT_6619 53/133, PAT_6704 38/128, PAT_6854 132/175, PAT_6953 131/147.
- Excluded whole patients: EL044 (ECoG only, no depth), PAT_6684 (seizure during the task){{, EL031 (its audio block has 23 trials with the patient asleep, 8 survive the filters; processed with a whole-recording CAR for a look on 2026-10-09)}}; PAT_3301 (only picture on disk) contributes nothing.
- Sampling: Bern 1024 Hz (h5 / edf), Geneva 2048 Hz (Micromed TRC), MicroEPI 2048 Hz (mat export with macro + microwire channels).
-2 Gate statistics per patient on v11: 6.3 % (PAT_5515) to 98.9 % (EL045), with PAT_3415 at 100 % of the 24 depth contacts it has left. In v9 five patients sat at 100 % because a condition had only 8–24 trials; on v11 the lowest count anywhere is 20 and only EL045, EL043 (91.0 %) and PAT_6953 (89.1 %) are near the ceiling.
-! Gate statistics per patient on v15: 6.3 % (PAT_5515) to 95.3 % (EL043); PAT_3780 82 % (92 of 112) once its trials sit on the stimuli; the lowest trial count in a condition is 20.
-4 Demographics table: age, sex, handedness, language, hemisphere of implantation, seizure focus — from the clinical files [your slot].
## R2 · One electrode, one representation (FIG 0A, FIG P.2–P.4)
- Product per electrode × condition: 103 frequency bins (0–398.4 Hz, 3.906 Hz step) × 300 time-normalised bins, dB relative to the pre-stimulus baseline; stimulus = bins 0–149, response period = bins 150–299, the response cue at bin 150.
- Odd / even trial halves stored beside every cube (used for reliability and for the noise-scaled gate).
- Example electrodes on the figure: auditory (PAT_3455 OTD7, STG), visual (EL043 IOG7), motor / production (PAT_6953 TPD10), preparatory (PAT_6854 IAG6).
-2 What the dB scale shows: broadband high-frequency increases and low-frequency decreases locked to stimulus and to the cue.
## R3 · The activity gate and the cohort (FIG S0)
- Gate per electrode-condition: ≥ 2 % of bins above +2.2 dB or ≥ 4 % below −3.0 dB; an electrode enters if any condition passes.
- Noise-scaled gate, introduced in v10 and unchanged in v11: a bin counts only beyond max(2.2 dB, 2 × noise) / min(−3 dB, −2 × noise), noise = sd(half1 − half2)/2 of that electrode-condition (the noise of the trial average, ~1/√N). On the v9 cubes it took 1695 → 1620 electrodes. The 0.2 s floor puts more trials into each average, so on v11 the noise itself fell: median 0.92 → 0.89 dB, worst 2.52 → 1.61 dB.
- v11 cache: 9825 electrode-conditions, noise median 0.89 dB, thresholds raised for 625 rows (largest ±3.2 dB), 2795 rows pass (v10: 9345 rows, 1198 raised, largest ±5.0 dB, 3173 pass; v9: 3456 of 9432); halves present for every row.
-! v15 cache, the 29 cohort patients: 8928 electrode-conditions, noise median 0.90 dB (worst 1.44), thresholds raised for 531 rows, 2985 rows pass.
- Why: with a fixed threshold the gated fraction followed the trial count (Spearman −0.79 across patients); split-half Pearson r of the whole cube was too weak a criterion (r ≥ 0.05 leaves 1446).
-2 Trials lost upstream on v11: 4740 trials in → 3777 used (79.7 %). The post-offset floor was lowered from 1 s to 0.2 s on 2026-09-26 — 1 s was too aggressive; on v9 it alone had cost 760 trials (4405 → 3217, 73.0 %). {{On v15 (29 cohort patients): 4580 in → 3638 used (79.4 %), 20–51 per condition and patient.}}
-3 Report the gate as a coverage statement, not a selection of responders: FIG 0B draws kept vs rejected electrodes on fsaverage.
## R4 · Clustering the responses (FIG 1a–d)
- Same electrodes in every run (v8: 1680; v9: 1695; v10: 1624; v11: 1487{{; v15: 1613, all twelve runs of 2026-10-09}}), four feature sets: `concat_hg` (mean of the 21 bins 70–148 Hz per time bin × 300 × 3 = 900 features), `concat_rawds` (15 bands 1–400 Hz × 30 time bins × 3 = 1350), `concat_bands5` (5 bands 1–20 / 20–70 / 70–170 / 170–270 / 270–400 Hz × 30 × 3 = 450), `concat_bands5z` (bands5 z-scored per band across the whole cohort, units = SD).
- Time downsampling 300 → 30 bins per condition block with scikit-image `resize` (anti-aliased), never across a block boundary.
- Three algorithms: convex NMF (graded loadings), k-means, Ward agglomerative; all fitted in unit-norm space (each electrode's feature vector scaled to unit L2 length so distance measures shape, not amplitude).
- K = 5…30 swept for every (algorithm, feature set); bi-cross-validated held-out variance peaks only for convex NMF (v9: K = 10 / 13 / 13 / 14 for hg / rawds / bands5 / bands5z); k-means and Ward curves rise to the edge of the range (0.36 at K = 8 → 0.48 at K = 30). {{The v15 fits were made without the bi-cross-validation (`--no-heldout`), so no held-out curve or peak exists on v15 yet; the peaks quoted are v9's.}}
-2 At the held-out peaks, 15–45 % of electrodes sit in clusters that are > 50 % one patient (bands5 at K = 14: one cluster = 75 of one patient's 77 electrodes). At K = 8: 0–10 %.
-2 Published cut: K = 8 (largest K with no one-patient cluster on HFA); cluster sizes on concat_hg cNMF K = 8 (v7 run): 248, 321, 159, 236, 216, 203, 126, 210 (largest 18.7 %).
-2 Graded membership: on concat_hg (v9, K = 7 publish fit) 71.6 % of electrodes have no majority component (top weight < 0.5), 1.1 % are dominated (≥ 0.8), median top weight 0.40 — the argument for the graded description.
-4 Decide the paper's K: 8 for every cross-method figure, or the cNMF peak per feature set (10 / 13 / 13 / 14) as 249 uses for the statistics.
-3 Named types (current figure vocabulary): production (rise after the cue in every condition), auditory onset, ramp to cue, suppression (below baseline during the stimulus), reading-only / visual.
-4 Per-cluster numbers to quote at the final K on v11: size, patients per cluster (9–21 on v7), largest patient's share (median 29 %), condition profiles.
## R5 · What survives the analysis choices (FIG 2)
- Correspondence by Hungarian assignment on the correlation of mean-centred loadings (shared-electrode counts when a side is a hard partition); reference solution = convex NMF on concat_bands5; Jaccard of the hard labels per matched pair.
- FIG 2 (built by `00_paper2_figures2_2.py`, K = 8): top = cNMF on the four feature sets, bottom = cNMF / k-means / Ward on concat_hg; mean pairwise ARI at every K 5–30, full ARI and NMI matrices, per-electrode modal agreement after matching (drawn on brains) with a 200-permutation null; per-cluster Jaccard vs reference with its own null; self-stability (Jaccard over 50 refits on 80 % subsamples) and PAC.
-2 Numbers on v8: two algorithms on one feature set agree on ~60 % of electrodes; two feature sets under one algorithm on ~40 % (chance 12 %); 31 % of electrodes are placed identically by all four feature sets (chance 0.4 %); ARI k-means–Ward 0.40, k-means–cNMF 0.34, Ward–cNMF 0.29 on concat_hg at K = 10 (v6 statistics).
- FIG 4 (`00_paper2_figure4_correspondence.py`, K = 8, 1000 permutations): every solution described in the shared concat_bands5z space, centroids matched greedily by Pearson r, overlap as Jaccard and weighted Ruzicka, nulls by plain and within-patient reassignment; FIG 5: Hungarian diagonal share of the contingency table for 9 solutions vs 1000 label shuffles. {{On v15 (2026-10-09, concat_hg, K 6–12) the three algorithms share 51–64 % of the diagonal against 8–21 % for random pairings (z 2.8–9.6); the FIG 4 / 5 figure set itself still has to be regenerated on the v15 runs.}}
-3 The auditory-onset type is recovered by every method; the weakest clusters have no counterpart (Jaccard ≤ 0.1).
-4 FIG 2 on disk is v6–v8 and the FIG 4 / 5 tables are v10; rebuild on v11 before quoting any number.
## R6 · Where the types sit (FIG 4, LanA)
-2 Language-atlas (LanA, 806-subject probabilistic atlas) proximity predicts more HFA while the prompt is heard and during production, less during visual encoding; |ρ| ≤ 0.16, FDR-cleared because n = 2644 (August contact set).
- Per-type test (`00_paper2_figure3_lana.py`, concat_hg K = 8): Spearman ρ between each cluster's loading and the LanA probability at the electrode; two nulls of 1000 permutations (labels rolled along each shaft; within-patient shuffle), 1000 patient bootstraps for the CI, Benjamini–Hochberg FDR, electrodes with atlas coverage ≥ 0.70 only.
-3 Current draft on the site: 0 of 8 types above the shaft-shift null — recompute on v11.
-4 Decide: is the anatomy result a Results section or a Limitations paragraph?
## R7 · Quality control that shaped the cohort
- Per-patient audit of the ERSP run (141): channels in / neural / Unknown-dropped / aux, bad list, reference route and WM count, notch per condition, trials in / kept, cube counts vs the previous tree. {{Since 2026-10-12 it reads the newest run that FINISHED ("Pipeline done" in the log), not the newest log: an interrupted 140 had made it describe PAT_3975 by two aborted attempts instead of the run its cubes came from.}}
- Review mode of the LM visualiser: every contact of every patient as a concatenated ERSP tile with its native tissue label, status (data / WM reference / bad / Unknown / not recorded) and split-half r, stripe index, HG.
- Stale-cube sweep (142): files older than the run start are deleted so a rerun cannot leave orphan cubes in the cache (v8 had carried 48).
- Trigger census against the photodiode: every logged trial of EL037, EL040, EL042, EL045, EL046, EL052 sits on a pulse; EL038's tables rebuilt from the full session log (14 audio + 2 reading trials recovered, 2026-09-24).
- Per-contact review of the noisiest patients (2026-09-28): 70–150 Hz floor of every channel against the patient's median, ERSP and HG raster per candidate; a reference contact 11 dB above the floor (EL045 PlaT_L4), a contact dead in two blocks (EL037 pH_R12) and a noisy shaft segment (EL043 sSMG8–10) removed; the bad list now applies before the reference and the notch.
- White-matter candidates of EL051 / EL052 checked one by one (each against the other candidates) before their BIDS tables replaced the Lookup workbooks: one responsive contact kept as data (EL051 POp_R1), one artefact contact removed (EL052 Pul_R16).
-! PAT_3780's trial times were wrong by 180 s (found 2026-10-07): the 2025 table was the stimulus log minus 30 s, the photodiode says log plus 150 s. The diode is hidden in two saturated inputs: envelope(X1) − envelope(X2), low-passed at 20 Hz, gives a −10…−20 µV step at every onset and the mirror step at every offset. 149 of 150 trials matched, tables rebuilt (`pat_3780_build_triggers.py`), FAP9 / HAG1 / HAG2 / TPG1 / TPG2 bad-listed, 140 rerun (112 cubes, 39 / 46 / 42 trials used); the evoked responses are now where they belong (audio: TOG1–4, |t| 10–12 at ~210 ms). Everything computed on the old windows for this patient is void.
-! PAT_3975 checked the same way (2026-10-09): E1 − E2 is a clean diode, its 2023 tables are right; 22 / 11 / 17 channels with |t| > 5 at 90–320 ms.
-! EL031 (2026-10-07 → 09): three one-block EDFs joined (`RAW_CONCAT`; the 20 March shaft renamed antSUP → antSFG by the new `RAW_CONCAT_RENAME`), DC6 triggers per file (`el031_build_triggers.py`: 51 / 23 / 53 trials), shafts kept by prefix (`EL_DEPTH_KEEP_PREFIXES`), Lookup anatomy, a whole-recording CAR for a look; out of the cohort: the audio block has 23 trials with the patient asleep, 8 survive the filters.
-! Contact review of 2026-10-12 (148 + the figures): every contact of every patient measured on the WM-referenced signal as a multiple of its own montage median; the worst 8 per patient and the whole of PAT_3975's reference laid out on the s1 tab as clean ERSP + HFA raster + Signal traces (232 figures, `make_contact_review_block.py`), one overview PDF per patient (`01_FBM_LM_report.py`: PSD per shaft, then ERSP / HFA / Signal per contact). Verdicts contact by contact, with the measurement on record when it argued the other way (PAT_3455's CPD9–12 are its deepest and quietest contacts and went out on the traces; CPD1 / TOD1 / TSP1 measure 1.0 / 0.8 / 0.6× and stayed, so "contact 1 is noisy" is not a rule).
-! The stale-cube sweep earned its keep on 2026-10-09: PAT_3780's four removed shafts had kept their cubes and halves through the rerun; `142_stale_cubes.py --delete` took them out before v15 was built. The readable part of 141 had broken when the 04 tree was archived (a column left in the header); fixed. A Word report of the preprocessing changes 09-28 → 10-09 (references, bad lists, trials, the three patients) is in `01_FBM_Analysis/outputs/preprocessing_docs/`.

# Discussion
-3 Response types are a property of what an electrode does, not where it sits (the through-line).
-3 K is a judgement between fit and generalisation: held-out variance cannot see a cluster that is one patient.
-3 Hard partitions of these data are not reproducible across preprocessing; the graded decomposition is the analysis, an argmax map is only shown beside its loadings.
-4 Relation to ESM: a passive map can report a type per electrode, not only active / silent [cite: Author et al., sec _ par _].
-4 Relation to language models of cortex (dual-stream, LanA): [cite: Fedorenko et al.; Hickok & Poeppel — sec _ par _].
-4 What the stimulus-locked vs cue-locked types say about production vs comprehension.

# Limitations
-2 sEEG coverage is patient-specific: a cluster should draw on fewer patients than a random draw; patients per cluster 9–21 vs 25.9 of 27 at random — a v8 statistic on 27 patients, to be recomputed on v11's 30.
-2 Trial counts differ (20–51 per condition after filters on v11{{ and on v15}}; 8–51 while the floor was 1 s); the noise-scaled gate corrects the noise dependence, not the estimate quality.
-2 The time axis is warped (proportions 0 / 0.5 / 0.5): no absolute latencies; the fixation period is not in the cube.
-3 Three Bern patients (EL036, EL037, EL040) carried trial-locked mains stripes in earlier trees; the per-shaft spectrum interpolation was introduced for them — state what the current tree still carries.
-3 White-matter references are thin for some patients (EL051 3 contacts, EL045 4, EL036 4; 3–53 across the cohort): a reference of few contacts carries more of its own noise into every channel. {{Since 2026-10-12 PAT_3975 runs on 11 deep-WM contacts (was 48 gyral ones next to cortex) and PAT_3455 on 11 (was 20).}}
-4 Language and site heterogeneity (German / French, Bern / Geneva) [your slot].

# STAR Methods · resources
## Key resources table (software)
- Python 3.11 (Windows); numpy (arrays, FFT, warp); scipy.signal (spectrogram, welch, iirnotch, filtfilt, resample_poly, butter, hilbert, savgol_filter, find_peaks); pandas (+ pyarrow for parquet, openpyxl for the Lookup workbooks); h5py + hdf5plugin (Bern h5, MicroEPI mat v7.3); neo MicromedIO (TRC); mne (read_raw_edf only in stage 01; Brain / surfaces in stage 02); matplotlib (all figures); scikit-learn (k-means, Ward); nibabel (FreeSurfer surfaces, annots, mgz); scipy.spatial cKDTree (vertex lookups); pyvista / VTK (off-screen renders); imageio; Niivue 0.45 / 0.69 (web pages).
- scikit-learn: KMeans (240, cNMF initialisation, consensus, nulls), AgglomerativeClustering (bi-CV), KNeighborsClassifier(1) predictor, PCA(2), silhouette / Calinski–Harabasz / Davies–Bouldin, ARI, NMI; scipy: cluster.hierarchy (Ward in 241), spatial (pdist, cKDTree), optimize (nnls, linear_sum_assignment), stats (spearmanr, t); scikit-image resize; joblib; convex NMF in-house (`functions/lf_decompose.py`).
-4 Versions: not pinned in the repo; read them off the environment before submission (`pip freeze`).
- FreeSurfer recon-all / FastSurfer outputs (elec_recon/*.electrodeNames, *.LEPTOVOX, surf/?h.pial, ?h.sphere.reg, mri/transforms/talairach.xfm); fsaverage; Yeo 2011 7/17-network annots; Desikan-Killiany aparc; Destrieux a2009s for region outlines.
-3 LanA language atlas (Lipkin et al. 2022, 806 subjects) [cite: Lipkin et al. 2022, Sci Data — sec _ par _].
- Code: `lorafanda/Analysis_LoraFanda` (stages 01 preprocessing, 02 clustering / recon); status site and visualisers at `lorafanda.github.io`. {{The whole chain for a new cohort is one flat list of commands in `02_FBM_Clustering/RUNBOOK_new_cohort.md` (2026-10-12): 140 → 141 → 142 → 143 → 147 / 148 → the cache → the fits → `make_paper2_outputs.py` phases.}}
## Resource availability
-4 Lead contact, materials availability, data and code availability statements [your slot].

# STAR Methods · participants and recordings
-2 Patients with drug-resistant focal epilepsy implanted for presurgical evaluation at Inselspital Bern (EL) and Geneva University Hospitals (PAT, MicroEPI); informed consent; ethics approvals [cite: approval numbers].
- Cohorts and files: Bern .h5 (Blosc-compressed from EL052 on) or .edf, 1024 Hz; Geneva Micromed .TRC, 2048 Hz; MicroEPI .mat exports (dataEcog macros, dataMicroDown microwires, photodiode), 2048 Hz; split recordings joined for EL043 and EL051 (RAW_CONCAT){{, and EL031's three one-block EDFs (RAW_CONCAT with RAW_CONCAT_RENAME, 2026-10-07)}}.
- Electrode types: sEEG depth contacts throughout; EL044 subdural grids only (excluded); PAT_3415 mixed (64 grid contacts dropped, 57 depth kept{{ — in the 03_ERSP run only 29 depth contacts survive, all IMG / IPG, which the noisy-shaft rule drops, so PAT_3415 has 0 electrodes in v15; the Unknown drop of OI / OS / TM / TP is to be looked at}}); MicroEPI Behnke-Fried style shafts with microwires (microwires removed from the cohort by the lower-case-m rule: `ADm1`, `X1M`…).
-4 Recording hardware, amplifier filters, reference at acquisition [your slot; not in the code].
-4 n patients / n electrodes / n per site in the final cohort [read off v11]. {{On v15: 29 patients / 1613 electrodes; 15 Bern (EL) and 14 Geneva (PAT), of which 5 are the MicroEPI patients.}}

# STAR Methods · task
-2 Language mapping task (paper 1): three blocks (auditory definition naming, picture naming, sentence completion by reading), 53 / 54 / 53 trials, randomised exemplars with a few repeats, spoken responses; PsychoPy log with onset, duration (= response latency from stimulus offset), response_type (correct / wrong), exemplar, epoch timestamp.
- Trial timing from a photodiode on DC6 (Bern) / trigger inputs (Geneva): stimulus onset = pulse rising edge, stimulus offset = falling edge; trial_end = offset + logged latency × fs.
- Photodiode detection (`LFfunctions_PDextract`): 10 Hz 2nd-order Butterworth low-pass (zero-phase), Savitzky–Golay derivative (65.5 ms window, order 5), peaks of the normalised derivative ≥ 0.40 at ≥ 0.2 s spacing; offsets = steepest falling edge before the next onset; per-patient flip for inverted diodes; manual trigger tables for EL030, EL038, EL051, EL052, PAT_3301, PAT_3975, PAT_3780. {{Since 2026-10-07 PAT_3780's tables come from its own builder (the X1 / X2 envelope difference, 4-MAD edges, one lag fitted over all onsets, offsets from the design durations), EL031's from the DC6 pulses of each of its three files.}}
-! The trigger channel itself is checked (`make_photodiode_figures.py`, 2026-10-12: the whole session with the channel's rail drawn in, and sixty seconds with the prep0 onsets on top). PAT_3455's diode rails 37 % of the time and, wired against ground, puts a −882 µV step on all 116 contacts at every onset — identical everywhere, so the white-matter reference removes it and only its spread between contacts survives; PAT_3975's rails 31 % and leaks into none of its 218 contacts. Saturation on its own is harmless; the step on the brain channels is the number to watch.
- Trial tables: one .tsv per condition in prep0 (`onset, onset_duration, sample, sample_offsets, trial_end, condition_name, resp_accuracy, trial_idx`); byte-identical duplicates skipped by MD5.
-3 Stimulus lengths: pictures 1.0 s, written sentences 3.5 s, audio 3.2–7.2 s (the sound file); identical per exemplar across patients (validated to ≤ 20 ms).
-4 Stimulus material source and language versions (GER / FR) [cite: paper 1, sec _ par _].

# STAR Methods · electrode localisation
- Native: contacts in LEPTOVOX voxel coordinates → surface RAS (tkrRAS) with the mgz vox2ras_tkr (nibabel), k axis flipped; distance to the nearest native pial vertex (cKDTree).
- WM vs cortical: BIDS electrodes.tsv `tissueLabel` first token `wm-` and `tissueWeights_1` > 0.97 → WM (the docstrings say "= 1.0"); the same table for EL051 / EL052 since their FreeSurfer reconstruction (2026-09-28; the Lookup workbook route is retired).
- To fsaverage: cortical contacts by spherical registration (nearest native pial vertex → subject ?h.sphere.reg → nearest fsaverage sphere vertex → its pial coordinate; snapped to the surface); WM contacts and patients without sphere.reg by affine (tkrRAS → scanner → MNI305 via talairach.xfm → fsaverage).
- EL051 / EL052: on their own FreeSurfer reconstruction since 2026-09-28 (spherical route like every Bern patient; until then Lookup MNI coordinates had served as fsaverage RAS, ≤ ~2 mm off); recording and recon names reconciled, 100 % matched.
- Labels: Yeo-7 / Yeo-17 at the nearest fsaverage pial vertex; Desikan-Killiany aparc at the nearest vertex; WM rows labelled "WhiteMatter". [`aparc_lookup.csv` predates PAT_1327 / EL051 / EL052 — rerun 211 before quoting their labels.]
- Name reconciliation between recording and reconstruction (RECON_ALIAS: EL034 MFG→MFG-L, EL043 pl→pIns, EL045 PlanTL→PlaT_L, EL046 al_L/pl_L→aI_L/pI_L, PAT_6619 OFAD/OFPD→OFA/OFP, EL051 ANT_R/aSMG_R/aSPL_R/pSPL_R/pSTG_R/Pul_R/VIM_R→ names without the hemisphere, EL052 al_R/ml_L/ml_R/PreCG→aI_R/mI_L/mI_R/PrCG_R) and between recording and BIDS (CHANNEL_SHAFT_ALIAS, the same maps inverted; 137 contacts in 10 patients on 2026-09-22, EL051 / EL052 on 09-29).
-4 CT–MRI co-registration and contact localisation software (iELVis? Lead-DBS? in-house) and FreeSurfer / FastSurfer versions [your slot; not in the code].
-3 Subcortical contacts (hippocampus, amygdala) are not `wm-` labelled, so they take the surface route and are snapped to the pial surface — decide whether to keep them at volume position for the anatomy figures.

# STAR Methods · preprocessing (stage 01, `140_ersp_pipeline.py`)
## Channel selection
- Auxiliary channels removed by name (MRK, MKR, X, ECG, EX, AUDIO prefixes; the export stage also drops PHOTO, EKG, EMG, EOG, TRIG, PULSE, RESP, AINP, E1–E8, X1–X8).
- Contacts without a parcellation (tissueLabel empty / "Unknown") dropped in every cohort (since 2026-09-23; 29 MicroEPI contacts).
- Bad contacts from a per-patient manual list (`bad_channels_manual`, visual / clinical review: per-contact sheets of floor vs the patient's median, ERSP and HG raster) are dropped right after the parcellation filter, before the reference, the notch and every figure (since 2026-09-28; until then they were only kept out of the reference mean and the export, and still shaped the shaft median the notch decides on).
-! Where a noisy contact is visible (2026-10-12, `148_contact_noise_audit.py`): only in the white-matter-referenced time domain. The cubes cannot show it (dB re baseline divides out a contact that is equally noisy in baseline and trial — PAT_3455's IPD14 scored 0.91 there while being one of the hairiest traces of its shaft), the raw cannot (before re-referencing every channel reads the same, the shared reference dominates: PAT_3455 sd 957 µV on all 116 channels), and the trial scores cannot (`score_hg` is a within-channel z, every contact sits at 1.00×). The audit measures broadband and 70–150 Hz sd per contact after subtracting the WM mean, over the task window, as a multiple of that patient's own montage median (`outputs/contact_noise/<pid>_contact_noise.tsv`). No absolute threshold: broken contacts separate by an order of magnitude (PAT_3965 cmd11 / cmd12 at 28×, its connector block at 15–18×, then nothing until 2.5×) while PAT_3455's worst is 3.4×; a contact is removed when it stands apart from its own montage.
-! Bad lists of 2026-10-12 (67 contacts, 10 patients): EL030 aH_L11; EL034 aH-L7; EL036 A_R9–11; EL042 EntG_R4; EL046 pI_L2; PAT_1327 HAG2, AG8; PAT_3455 OTD8 (a reference contact with a task response of its own, +0.56 dB) + seven shaft-entry contacts at 1.7–3.4× + CPD2 / CPD9–12 / OPD5 / TSP4–6 (nine members of its old 20-contact reference, so it was rerun: reference 20 → 11); PAT_3965 had2 (5.9×); PAT_3975 39 contacts (TPG / TPD whole, AD1–10, AG1–6, HAD5–8, PHG1–2, IAG1–4, IMD16–18: amygdala, ventricle, cortex and gyral WM by the anatomy table, 1.4–4.7× broadband); PAT_6704 FPD6; PAT_6854 IMG1 + eleven more. Each config entry says what measured what.
-! Removing a contact after its patient was run: `143_delete_non_neural.py --bad-listed --delete` takes out its cubes, halves, images and trial-score rows without a rerun and writes `logs/bad_listed_deleted_<stamp>.tsv`, which the s1 per-patient table reads — unless the contact was in the reference mean, in which case the patient is rerun (PAT_3455, PAT_3975, PAT_6854 on 10-12). `147_keep_all_channels.py` runs 140 with the bad list cleared into `outputs/03_ERSP-allchan-wm` (`--keep-wm` also keeps the reference contacts as data, `--fast` writes Signal / HFA / trial scores only): the only tree where a removed or reference contact can be looked at.
- Grid contacts (EL044 P / Pa / T / postP; PAT_3415 GA–GH) and microwires (`…m1`, FOM, X1M) never enter the clustering cohort.
- Bern recordings cropped to the task window (`time_range` per patient); EL043's hemisphere suffix stripped to match its reconstruction.
## Re-referencing
- One reference per patient, computed on the whole cropped recording before notching and epoching, subtracted from every channel.
- White-matter reference: mean of ≥ 3 WM contacts (rule above, from the BIDS electrodes table of every patient), bad-listed WM contacts left out; WM reference contacts are not analysed as data (Bern / Geneva) — MicroEPI keeps them as data (selective reference, macros only); `WM_NOT_REFERENCE` keeps a WM contact with a task response as data: PAT_6704 THD1 / THD3 / THD4, EL051 POp_R1 (+0.5 dB high gamma through the response half).
- Reference size 3 (EL051) to 53 (PAT_3301) contacts; each candidate of EL051 / EL052 was checked against the mean of the others (ERSP, high gamma within ±0.45 dB, broadband floor) before the switch from their Lookup workbooks (2026-09-29). No whole-recording CAR remains (EL052 had one while its workbook listed no plugged WM contact); EL044 per-grid CAR (P 6, Pa 55, T 48, postP 15).
-! The rule reaches gyral white matter only (found 2026-10-12): `derive_wm_channels_from_electrodes_tsv` keeps `tissueLabel` tokens of the form `wm-<hemi>-<parcel>`, the core under a gyrus's ribbon; deep white matter is labelled `Left-/Right-UnsegmentedWhiteMatter` and never qualifies. For PAT_3975 that gave a 48-contact reference a median 3.5 mm from the nearest grey contact, carrying 0.95× the broadband and 0.85× the high-gamma amplitude of the grey contacts with the same dB modulation — not a silent baseline (not the dB normalisation: absolute amplitude says so; not a shared-reference artefact: between-shaft cube correlation +0.17). PAT_3975 now runs on its 11 Unsegmented contacts (`cfg.MANUAL_WM_CHANNELS`: CPD2, CPG3–7, FOG3, IAD9, IAG8, IMD9, IMD10), all quieter than the montage median. PAT_3455's reference is 11 contacts since the same day (HPD5, HPD7, IMD3–5, IMD11, IPD8, IPD14, PHD5, PHD6, TOD6); EL031 is the one whole-recording CAR (`WHOLE_CAR_PATIENTS`).
-3 [cite: WM referencing in sEEG — Arnulfo et al. 2015, J Neurosci Methods 240:1; Mercier et al. 2017, NeuroImage 147:219 — sec _ par _]
-4 Reference counts per patient: 4–53 WM contacts (audit table) [pick the range to quote].
## Line noise
- Per condition block (trials ± 10 s), on the block's own median Welch PSD (2 s segments, 0.5 Hz): candidate 50 Hz harmonics to 400 Hz, plus a second comb where the unexplained-peak audit showed one (EL048: 16.667 Hz railway traction; EL045: 59.94 Hz, the screen's refresh, visible once its noisy reference contact was removed); a harmonic is corrected when its peak z ≥ 3 against the ±1–5 Hz surround.
- One method since 2026-09-24: spectrum interpolation for every patient, decided per shaft (amplitude inside the measured peak width replaced by the flanks' RMS with random phase, width widened +1 Hz per pass while the ring 3 Hz outside stands > 3 dB above the far floor, up to ±12 Hz). `notch_method_default = "interp"` and `notch_shaft_all = True` in config; the audit reads `interp` / `per shaft` for all 32 patients. Before this most patients got an IIR notch (`scipy.signal.iirnotch`, Q from the peak's sharpness, zero-phase), which dug holes of −0.7 to −8.0 dB into the spectrum.
- The widening test reads the time-averaged PSD, so a residue that is trial-locked but small on average passes it and still stripes the ERSP row (EL037: the 350 Hz row 1.15 dB below its neighbours on every trial with the band at ±1 Hz). A per-patient floor on the starting half-width (`notch_interp_min_hw_hz`: EL037 and EL045, 350 Hz, ±5 Hz) removes it (row bias +0.09 after); the check is the harmonic row's bias against its ±2 neighbours in the new cubes.
- Audit per harmonic: before / after dB against the floor, width used and residual hole width; unexplained-peak table after the notch.
-2 [cite: spectrum interpolation — Leske & Dalal 2019, NeuroImage 189:763, sec 2.3] ; FieldTrip `dftfilter` with `dftreplace = neighbour` is the same idea.
- No other filtering: no high-pass, no low-pass beyond the anti-alias of the resampling, no detrending.
## Trial selection
- A trial is kept when the response is scored correct, the stimulus lasted ≥ 0.5 s, the post-offset period (offset → trial_end) is 0.2–10 s, and it is inside the Tukey fences (1.5 × IQR) of the condition's post durations; trials overlapping a listed discharge span are dropped (none in the current cohort).
- Per-trial ERSP rejection exists but is off for every current patient.
- Yield: 3777 of 4740 trials (79.7 %, v11 cohort); per condition 20–51 per patient.
-3 [cite: Tukey fences / IQR outlier rule — Tukey 1977, sec _]
## Time–frequency and the ERSP cube
- Each channel resampled to 1 kHz (`resample_poly`, 125/128 or 125/256).
- STFT (`scipy.signal.spectrogram`, Hann, nperseg 128, noverlap 108 = 84 %, nfft 256 → 3.906 Hz bins, 20 ms hop), power in dB; bins ≤ 400 Hz kept (103).
- Baseline per trial and frequency: mean dB over frames centred in −0.4 to −0.1 s before stimulus onset (config and 140 agree since 2026-09-24; the trial segment itself starts at −0.6 s); `Srel = S_dB − baseline`.
- Time normalisation: stimulus onset → offset interpolated onto bins 0–149, offset → trial_end onto 150–299 (proportions 0 / 0.5 / 0.5); the pre-stimulus period is not in the cube.
- Average over kept trials (nanmean of dB), NaNs filled by nearest neighbour; halves = odd / even kept trials, not filled.
- Output: `ERSP_matrix/<cond>/<pid>_<cond>_<ref>_ERSP_<contact>_TN.npy` (103 × 300 float), halves, a CLEAN png; QC ERSP and HG trial rasters for every channel.
- High-gamma QC raster: 70–150 Hz 4th-order Butterworth (zero-phase), Hilbert envelope, 25 ms boxcar, z-scored to the trial's own −0.6…−0.1 s window; trials sorted by stimulus duration, excluded trials drawn last with their reason.
-3 [cite: ERSP — Makeig 1993, Electroencephalogr Clin Neurophysiol 86:283, sec _]
-3 [cite: time normalisation / warping of variable-length trials to proportions — pick the paper you follow, sec _ par _]

# STAR Methods · dataset and gate (stage 02, `lf_dataset`, `lf_concat`)
- One sample = electrode × condition cube (103 × 300); the cache (`concat_source_v<N>`) records params, schema and every sample's file; a new cohort = a new version, never a rebuild in place (schema 3 refuses it).
- Filters at read: non-neural names, microelectrodes (`is_micro`: lower-case m before the contact number, plus FOM / X1M / X2M; PAT_ only), noisy shafts (PAT_3415 IMG / TA / IPG), grid shafts, excluded patients (EL044, PAT_6684{{, EL031 since 2026-10-09}}).
- Gate (above): fixed thresholds in v9, noise-scaled (k = 2) from v10 on; `high_activity` per electrode-condition, an electrode kept if any condition passes; proportions 2 % / 4 %.
- Cohort rule: all three conditions on disk, gated in ≥ 1, one row per electrode; concatenation audio | picture | reading → 103 × 900.
- Cohort sizes: v8 1680 / 27 patients; v9 1695 / 2966 / 28; v10 1624 / 2961 / 29; v11 1487 / 3121 / 30 (PAT_1327 added; the fall in the gated count is the re-preprocessed tree, not a rule change — the gate is the same one){{; v13 (09-29, the archived 04 tree) 1456 fitted; v14 (10-07, 03_ERSP, PAT_3780 still on the wrong windows) 1570; v15 (10-09) 1613 / 2976 / 29}}.
-2 Feature sets: `concat_hg` 70–150 Hz mean per bin (3 × 300); `concat_rawds` 15 bands × 30 bins per block; `concat_bands5` 5 bandwidth-weighted bands × 30 bins; `concat_bands5z` the same z-scored per band across the cohort.
-2 Normalisation: each electrode scaled to unit length before fitting (spherical k-means logic; without it added clusters split by amplitude: η² of norm 0.65 → 0.11, ARI 0.22 between the two spaces on concat_hg).
-2 [cite: Hamilton, Edwards & Chang 2018, Curr Biol 28:1860 (per-electrode normalisation before decomposition), Fig S2] [cite: Dhillon & Modha 2001, Machine Learning 42:143 (spherical k-means)]

# STAR Methods · clustering
- Convex NMF (Ding, Li & Jordan 2010, IEEE TPAMI 32:45), in-house NumPy implementation (`lf_decompose.convex_nmf`): X ≈ G (Wᵀ X) with multiplicative updates on the signed Gram matrix; initialised from k-means labels (n_init 10) with G = 0.2 (+1 at the assigned cluster) + 0.2 · U(0,1); stop when the relative Frobenius change < 1e-6 (checked every 10 iterations); W columns normalised to sum 1; loadings G row-normalised (graded membership), argmax only for display.
- Sweep (`sweep_decomposition.py`): K = 5…30, n_iter 1000, random_state 0, on the unit-norm rows of the newest k-means run's X_train; publish fit at K = 7 with n_iter 300 (`run_decomposition.py`).
- k-means: scikit-learn `KMeans(n_clusters=K, n_init=20, random_state=42)` (240); Ward: `scipy.cluster.hierarchy.linkage(method="ward")` on Euclidean `pdist`, cut with `fcluster(maxclust)` (241); both on unit-norm features; silhouette, Calinski–Harabasz and Davies–Bouldin recorded at every K (silhouette's best K is recorded, not used).
-! The v15 fits (2026-10-09) were made with the notebooks' terminal twins — `240_cluster_kmeans.py`, `241_cluster_hierarchical.py`, `242_cluster_cnmf.py`, generated from the notebooks by `scripts/notebooks_to_scripts.py` — with `--no-heldout`; same settings (K 5–30, n_init 20, unit-norm, cNMF n_iter 1000). The notebook kernels had died of a half-installed numpy 2.4.6 in LORA_ENV2 (a pip call from inside a notebook); numpy 1.26.4 was reinstalled clean. Rule since: no pip from a notebook, no kernel alive during an install.
- Every run labels the same electrodes in the same order; run manifests (params, git commit, Python version) under `outputs/clustering/<method>/<feature_set>/runs/<stamp>`; runs registered in `index.json`.
- K selection: bi-cross-validation (Owen & Perry 2009, Ann Appl Stat 3:564): 4 row folds × 4 column folds (seed 0), fit on train rows × train columns, held-out rows get loadings by NNLS (cNMF) or the nearest cluster mean (k-means / Ward), score = 1 − SSE/SS on the held-out block, 300 iterations; only convex NMF's curve turns over.
- Patient-generalisation criterion at every K: share of clusters where one patient holds > 50 % of the electrodes, and share of electrodes in such clusters (FIG 1D); per-cluster patient composition against 400 size-matched random draws.
- Stability (`measure_cluster_stability.py`): Hennig bootstrap Jaccard, 25 resamples, in-bag scoring, 95 % t-interval (lower bound > 0.75 stable, > 0.60 pattern), Gaussian-surrogate null (95th percentile); initialisation Jaccard over 10 seeds; persistence across K.
-2 [cite: Ben-Hur, Elisseeff & Guyon 2002, PSB 7:6; Lange et al. 2004, Neural Comput 16:1299; Hennig 2007, CSDA 52:258 — sec _ par _]
- Published K: 8 (the largest K with no one-patient cluster on concat_hg); the cNMF held-out peaks (v9: 10 / 13 / 13 / 14) are marked on the curve; 249 computes the statistics at those peaks.
- Matching across runs: Hungarian assignment (`scipy.optimize.linear_sum_assignment`; Kuhn 1955) on loading correlation, or on shared-electrode counts when a side is a hard partition; Jaccard on hard labels per pair; ARI (Hubert & Arabie 1985) and NMI between partitions.
- Cluster description: mean ± SD across the cluster's electrodes in the feature set's own representation (members = argmax); patients per cluster and the largest patient's share.
-3 Bi-CV fits k-means with n_init 10 and scikit-learn's Ward, the final runs with n_init 20 and scipy's Ward — same algorithms, different implementations; say "k-means with 10–20 restarts".

# STAR Methods · anatomy and atlas tests
-2 Electrodes rendered on fsaverage (pial, glass look, MNE `Brain` on pyvista, off-screen); per-cluster renders by patient and by condition (252).
-2 LanA: probabilistic language-network atlas sampled at each electrode's fsaverage coordinate; Spearman correlation between atlas probability and HFA per time window; FDR across windows.
-3 Spatial null for cluster–atlas enrichment: shafts permuted intact (keeps the spatial autocorrelation of an implant) — describe exactly as implemented in `04_FBM_Pooling/make_lana_runs.py`.
-4 [cite: Lipkin et al. 2022 (LanA); Fedorenko et al. 2010 (language localiser) — sec _ par _]

# STAR Methods · quantification and statistics
-2 Split-half reliability: Pearson r between the odd- and even-trial cubes (whole cube, 30 900 bins); reported per contact in review mode; whole-cube r is dominated by noise bins (median 0.15 among gated), which is why it is not the gate.
- Noise of the trial average: sd(half1 − half2)/2 (var(h1 − h2) = 4σ²/N → σ²/N for the full mean).
- Held-out variance ± 1 SD across bi-CV folds; generalisation curves at every K; patient composition vs random draw.
- 249 (`make_cluster_statistics.py`, at the cNMF peak K per feature set, N_NULL 50, LOPO_REPS 10): separation = silhouette in dB and unit-norm space against 50 Gaussian surrogates with the data's covariance (z-score); agreement = pairwise ARI / NMI and contingency tables between the three methods; anatomical coherence = share of the 10 nearest fsaverage neighbours sharing a label vs 20 label shuffles; leave-one-patient-out = ARI of the refit vs the full-cohort labels against size-matched pseudo-patient nulls (z of the minimum).
- Atlas test: Spearman ρ, 1000 shaft-shift and 1000 within-patient permutations, 1000 patient bootstraps, Benjamini–Hochberg FDR (FIG 3); correspondence nulls 1000 reassignments (FIG 4), 1000 label shuffles (FIG 5).
-4 Effect-size conventions and which of these tests reach the main text [your slot].

#+ STAR Methods · the pipeline, script by script
> Purple = new writing of 2026-10-13, to review. One script after the other, in the order they run; the numbers are the config as it stands (v16). Black / grey lines elsewhere in this tab carry the same facts with their sources.
##+ `140_ersp_pipeline.py` — one patient, raw recording → ERSP cubes
-+ Input: one raw recording per patient. Bern .h5 / .edf at 1024 Hz, Geneva Micromed .TRC at 2048 Hz, MicroEPI .mat at 2048 Hz. A recording split over several files is joined end to end (`RAW_CONCAT`); a shaft spelled differently in one file is renamed (`RAW_CONCAT_RENAME`).
-+ Bern recordings are cropped to the task window (`time_range` per patient).
-+ Channels are removed in this order: auxiliary channels by name (MRK, X, ECG, EX, AUDIO, photo, E1–E8); contacts with no parcellation (tissueLabel empty or Unknown); the patient's manual bad list (`bad_channels_manual`). A removed contact gets no product.
-+ Reference: the mean of the patient's white-matter contacts, taken from the BIDS electrodes table (`tissueLabel` starting with `wm-`, `tissueWeights_1` > 0.97, at least 3 contacts), computed on the whole cropped recording and subtracted from every channel. Reference contacts are not analysed as data (MicroEPI keeps them). Overrides: `MANUAL_WM_CHANNELS` (PAT_3975, 11 deep-WM contacts), `WM_NOT_REFERENCE` (a WM contact with a task response stays data), `WHOLE_CAR_PATIENTS` (EL031), a per-grid CAR for EL044.
-+ Line noise, per condition block (the block's trials ± 10 s): the block's median Welch PSD (2 s segments); the 50 Hz harmonics up to 400 Hz, plus a patient's own comb where the audit found one (EL048 16.667 Hz, EL045 59.94 Hz); a harmonic is corrected when its peak is z ≥ 3 against the ±1–5 Hz surround; correction by spectrum interpolation per shaft (the amplitude inside the peak replaced by the flanks' RMS with random phase, the width widened +1 Hz per pass up to ±12 Hz), with a per-patient floor on the starting half-width where a trial-locked residue needs it (`notch_interp_min_hw_hz`). No other filter is applied.
-+ Trials: the patient's prep0 tables, one .tsv per condition (onset, offset, trial_end, accuracy, exemplar); byte-identical duplicates skipped. A trial is kept when the answer is correct, the stimulus lasted ≥ 0.5 s, the post-offset period (offset → trial_end) is 0.2–10 s and inside the 1.5 × IQR fences of its condition, and it overlaps no listed discharge span.
-2+ Per-trial high-gamma rejection per channel: the 70–150 Hz envelope of each trial z-scored within the channel (`score_hg`); trials beyond 3.5 MAD are dropped for that channel.
-+ Each channel is resampled to 1 kHz (`resample_poly`). STFT per trial: Hann, nperseg 128, noverlap 108 (84 %, a 20 ms hop), nfft 256 → 3.906 Hz bins, kept to 400 Hz (103 bins); power in dB.
-+ Baseline per trial and frequency: the mean over −0.4 to −0.1 s before stimulus onset, subtracted (dB re baseline).
-+ Time normalisation: stimulus onset → offset onto bins 0–149, offset → trial_end onto bins 150–299 (proportions 0 / 0.5 / 0.5, 300 bins). The pre-stimulus period is not in the cube.
-+ The cube is the mean over the kept trials (nanmean, gaps filled by nearest neighbour). The odd and the even trials are averaged separately and stored beside it (`ERSP_halves`).
-+ Outputs under `outputs/03_ERSP/<pid>/LM/`: `ERSP_matrix/<cond>/<pid>_<cond>_WM_ERSP_<contact>_TN.npy` (103 × 300 float), `ERSP_halves/`, one CLEAN image per cube; the QC set per channel (ERSP image, HFA raster, Signal raster under `PerTrial/`), the trial scores, the raw and clean PSD per shaft, the IQR report, the run log.
-+ `--pd` first re-extracts the trigger times from the photodiode and rewrites prep0 (`LF_pd`: 10 Hz Butterworth low-pass, Savitzky–Golay derivative, peaks ≥ 0.40 at ≥ 0.2 s spacing, offsets = the steepest falling edge before the next onset, a per-patient flip). A diode that needs its own recipe has a builder: `pat_1327_build_triggers.py`, `pat_3780_build_triggers.py` (envelope difference of X1 and X2), `el031_build_triggers.py` (DC6 pulses per file).
##+ `141_audit_140.py` — the audit of the runs
-+ One row per patient, from the newest finished log and the config as it stands: channels in / neural / Unknown / aux, bad list, reference and its source, WM count, notch per block, trials in / kept / used, cubes and shapes. Writes `outputs/03_ERSP/audit_140.tsv`; the s1 tab reads it.
##+ `142_stale_cubes.py` — stale products
-+ Lists, and with `--delete` removes, every file under a patient's folder older than its run start. A contact a rerun no longer produces leaves no cube behind.
##+ `143_delete_non_neural.py --bad-listed` — a contact removed after its run
-+ Removes one contact's products (cubes, halves, images, trial-score rows) without a rerun and logs them (`outputs/03_ERSP/logs/bad_listed_deleted_<stamp>.tsv`). Not for a reference contact: the patient is rerun instead.
##+ `148_contact_noise_audit.py` / `147_keep_all_channels.py` — the contact audit
-+ 148: broadband and 70–150 Hz sd per contact after subtracting the white-matter mean, over the task window, as a multiple of the patient's montage median (`outputs/contact_noise/<pid>_contact_noise.tsv`). 147: 140 with the bad list cleared, into `outputs/03_ERSP-allchan-wm`, where a removed or reference contact can be looked at. The decision per contact is taken on the Signal, HFA and ERSP figures against the measurement (s1 tab, `make_contact_review_block.py`).
##+ `250_recon_fsaverage.ipynb` — electrode coordinates
-+ Contacts from LEPTOVOX voxels to surface RAS (tkrRAS); WM vs cortical from the BIDS table; cortical contacts to fsaverage by spherical registration (nearest native pial vertex → `?h.sphere.reg` → nearest fsaverage vertex → its pial coordinate), WM contacts and patients without `sphere.reg` by affine (`talairach.xfm` → MNI305 → fsaverage); Yeo-7 / Yeo-17 and Desikan-Killiany labels at the nearest vertex. Recording and reconstruction names reconciled (`RECON_ALIAS`, `CHANNEL_SHAFT_ALIAS`). Writes `outputs/250_recon/fsaverage/coords/<pid>_contacts_fsaverage.csv` and the ALL table.
##+ `rebuild_concat_cache.py --version N --apply` — the dataset cache (`lf_dataset.prepare_dataset`)
-+ Walks `ERSP_matrix` of every patient folder and stacks every cube into `X_3d.npy` (n × 103 × 300, float32) with `df_meta.parquet` (patient, electrode, condition, file).
-+ Per electrode and condition: noise = sd(half1 − half2) / 2 from `ERSP_halves`; a bin counts as active beyond max(+2.2 dB, 2 × noise) or below min(−3.0 dB, −2 × noise); `high_activity` when ≥ 2 % of the 30 900 bins are above or ≥ 4 % below.
-+ A new version directory every time the cohort changes (the params record no patient list). FIG C.0 is drawn from it (`make_cohort_figure.py`). v16: 8967 rows, 30 patient folders.
##+ `lf_concat.build_concat_dataset` — the cohort and the feature sets (called by 240 / 241 / 242)
-+ Filters at read: non-neural names; microwires (`is_micro`: a lower-case m before the contact number, FOM, X1M); grid shafts (EL044, PAT_3415 GA–GH); noisy shafts (PAT_3415 IMG / TA / IPG); excluded patients (EL044, PAT_6684, EL031).
-+ An electrode enters when it has a cube in all three conditions and `high_activity` in at least one. Its three cubes are concatenated audio | picture | reading (103 × 900). v16: 1607 electrodes of 2946, 29 patients.
-+ Feature sets from the concatenated cube: `concat_hg` (the mean of the 70–150 Hz bins per time bin, 3 × 300 = 900); `concat_rawds` (15 bands 1–400 Hz × 30 time bins per block = 1350; the time axis downsampled with scikit-image `resize`, never across a block boundary); `concat_bands5` (5 bands, 1–20 / 20–70 / 70–170 / 170–270 / 270–400 Hz, × 30 × 3 = 450; each band the mean of its original bins); `concat_bands5z` (the same, each band z-scored across the cohort).
-+ Every electrode's feature vector is scaled to unit L2 length before fitting (unit-norm space). `X_train.npy` keeps the raw dB.
##+ `240_cluster_kmeans.py` / `241_cluster_hierarchical.py` — hard partitions
-+ For each feature set: k-means (scikit-learn `KMeans`, n_init 20, random_state 42) and Ward (`scipy.cluster.hierarchy.linkage`, Euclidean, cut by `fcluster` maxclust) at every K from 5 to 30, on the unit-norm features. Per run: `labels.csv` (electrode order), `cluster_labels_by_k.csv`, `sweep_metrics.csv` (silhouette, Calinski–Harabasz, Davies–Bouldin per K), `X_train.npy`, a manifest naming the cache; one folder per fit under `outputs/clustering/<method>/<feature_set>/runs/<stamp>`.
-+ Section 3 of each (`make_heldout_variance.py`: bi-cross-validation, 4 row folds × 4 column folds, held-out electrodes scored by the nearest cluster mean, 300 iterations) runs only when K is being chosen; `--no-heldout` otherwise. Electrodes stay in the same order across every run.
##+ `242_cluster_cnmf.py` — convex NMF
-+ `run_decomposition.py`: convex NMF (Ding, Li & Jordan 2010; `lf_decompose.convex_nmf`), X ≈ G (Wᵀ X), multiplicative updates on the signed Gram matrix, initialised from k-means labels, stopped at a relative Frobenius change < 1e-6; the publish fit at K = 7 with 300 iterations; W columns normalised to sum 1, loadings G row-normalised (graded membership). The matrix is the newest k-means run's `X_train.npy`.
-+ `publish_decomposition.py` registers that fit as a run; `sweep_decomposition.py` refits K = 5…30 at 1000 iterations into it: `loadings_by_k`, `components_by_k`, `cluster_labels_by_k` (argmax, for display only).
##+ `make_missing_centroids.py` / `make_centroid_rasters.py` — per-run products
-+ The mean ± SD feature vector of every cluster at every K (`cluster_centroids/k_K/`, members = argmax, plus one panel at the run's K) and the electrode rasters of the run's K (`cluster_rasters/`).
##+ `choose_k.py` / `make_heldout_variance.py` — K
-+ K is read where convex NMF's bi-cross-validated held-out variance turns over; the k-means and Ward curves rise to the edge of the range. Beside it the patient-generalisation criterion at every K: the share of clusters where one patient holds > 50 % of the electrodes, and the share of electrodes in such clusters. The cross-method figures are cut at K = 8; the cluster visualizer carries K 5–12.
##+ `make_cluster_statistics.py` (249) — the statistics
-+ At the chosen K per feature set: separation = silhouette in dB and in unit-norm space against 50 Gaussian surrogates with the data's covariance; agreement = pairwise ARI / NMI and contingency tables between the three methods; anatomical coherence = the share of the 10 nearest fsaverage neighbours sharing a label against 20 label shuffles; leave-one-patient-out = ARI of the refit against the full-cohort labels, against size-matched pseudo-patient nulls. Writes `cluster_statistics.json` into the run. Not run on v16 yet.
##+ `00_paper2_figure0_coverage.py` — FIG 0
-+ Every electrode the gate saw, on fsaverage (kept vs rejected); four example electrodes with their three-condition cubes; the per-patient composition and the gate at the line as the supplement.
##+ `00_Paper2_Figures.py` — FIG 1a–d
-+ One feature set per figure, convex NMF: A the held-out variance vs K with the peaks and the cut; B one block per cluster, mean ± SD over the three concatenated conditions, hemisphere renders, patient-composition bar; C how to read a block (trial strip, GO cue at 50 %); D the share of one-patient clusters and of electrodes in them vs K. The renders need a screen (VTK).
##+ `00_paper2_figures2_2.py` — FIG 2
-+ Agreement at K = 8: top the four feature sets under convex NMF, bottom the three algorithms on concat_hg (run twice, the second with `--algo-feature-set concat_bands5z`). Solutions matched by Hungarian assignment on the correlation of mean-centred loadings (shared-electrode counts when a side is a hard partition), reference = convex NMF on concat_bands5. Mean pairwise ARI at every K 5–30, full ARI and NMI matrices, per-electrode modal agreement after matching with a 200-permutation null, per-cluster Jaccard against the reference with its own null, self-stability (Jaccard over 50 refits on 80 % subsamples) and PAC.
##+ `00_paper2_figure3_lana.py` — FIG 3
-+ Per cluster at K = 8 on concat_hg: Spearman ρ between the cluster's loading and the LanA probability at the electrode; two nulls of 1000 permutations (labels rolled along each shaft; within-patient shuffle), 1000 patient bootstraps for the CI, Benjamini–Hochberg FDR; electrodes with atlas coverage ≥ 0.70 only.
##+ `00_paper2_figure4_correspondence.py` / `00_paper2_figure4_panels.py` — FIG 4
-+ At each K: every solution described in the shared concat_bands5z space, centroids matched greedily by Pearson r, overlap as Jaccard and weighted Ruzicka, nulls of 1000 reassignments, plain and within-patient. The matched table (`FIG4_matched_K<K>_run<stamp>.csv`) is drawn as the panels.
##+ `00_paper2_figure5_confusion.py` — FIG 5
-+ Nine solutions (three algorithms × HFA / 15 bands / 5 bands z) against each other at each K: the Hungarian diagonal share of the contingency table against 1000 label shuffles.
##+ `make_paper2_outputs.py` — the order
-+ One command runs the rest in order: audit → select → runs → paper → stage02 → norm → prep → bundles → site (`--only` for phases, `--headless` leaves the five brain renders out, `--ks` the K values of FIG 4 / 5). It fits nothing: the cache and 240 / 241 / 242 come first. `02_FBM_Clustering/RUNBOOK_new_cohort.md` lists every command.

# Figures and legends
-2 FIG 0 · the task and the cohort: A four illustrative electrodes (auditory, visual, motor, preparatory) with their three-condition cubes on the ±7 dB scale; B every electrode the gate saw on fsaverage (kept green, rejected grey). Supplement S0: per-patient composition, atlas coverage, gate at the line.
-2 FIG 1a–d · one feature set each (hg, rawds, bands5, bands5z), convex NMF: A held-out variance vs K (all four curves, peak triangles, the cut K); B one block per cluster: mean ± SD over the three concatenated conditions, hemisphere renders, patient-composition bar; C how to read a cluster mean (trial strip, GO cue at 50 %); D generalisation vs K (share of one-patient clusters and of electrodes in them).
-2 FIG 2 · agreement (planned): feature-set agreement and algorithm agreement at a shared K; matched pairs and what is left out.
-2 FIG 4 · anatomy: cluster–atlas relation with the spatial null; LanA proximity maps.
-2 FIG P.2–P.4 (methods / supplement): reference and notch scheme; one trial → one ERSP; time normalisation.
-4 Graphical abstract: five types as traces + three one-number verdicts (K = 8; 31 % placed alike; 0 of 8 above the null) [decide].
-4 Figure numbering for Cell Reports (main figures ≤ 7; STAR Methods figures as S) [your slot].

# Check before the talk
- Baseline window — settled 2026-09-24: −0.4…−0.1 s is what every cube was computed with (the ERSPParams default); config now says so and 140 passes it explicitly; the s1 tab and this outline say −0.4 (the trial segment still starts at −0.6 s, which is what the site's −0.6 referred to).
- WM rule — docstrings, s1 tab and outline now say `tissueWeights_1 > 0.97` (the code was always that).
- STFT overlap — config comment fixed: 108 / 128 = 84 %, a 20 ms hop.
- `invalid_trials` — documented in config (they only colour the photodiode figure in the `--pd` path; `fake_trials` are removed); no table changed.
-3 `trial_idx` = exemplar id: `collect_trials` sorts by it, so odd / even halves interleave by exemplar, not by time (fine for reliability, say so).
-3 aparc cache built on pial for most patients and on white for EL051 / EL052; 13 patients have no aparc rows (EL044, EL046, EL048, PAT_6953 …) — rebuild before any aparc-based statement.
-3 The site's stage-02 result blocks were emptied on 2026-09-27 and only method and cohort text is left; the numbers come back when 240–242, 249 and 252 have run on v11. {{Still so on 2026-10-12: the fits exist on v15, the statistics do not.}}
- Runs on disk still mixed cohorts on 2026-09-27 08:03: Ward on all four feature sets was v11 (n = 1487); k-means and every convex NMF run were v10 (n = 1624). The statistics folders hold v6–v8 (n = 1688 / 1719 / 1680).
-! Since 2026-10-12 the tree is ahead of the cache again: PAT_6854 (13:53), PAT_3975 (15:34, deep-WM reference) and PAT_3455 (16:28) were rerun after v15 was built, and 67 contacts' products left the tree — nothing stage-02 is quotable until v16 exists and the twelve runs are refitted on it.
-! Runs on disk, 2026-10-09: all twelve are v15 (n = 1613), each tagged with its cache, so `lf_runs` accepts them and refuses everything older. FIG 0–3 are still v8 / v13 renders, FIG 4 / 5 of 10-09 are partial (the correspondence table exists at K 8 only; the panels were drawn from September tables), the held-out curves on the site are v13's, 249 and 252 have not run on v15. The cluster visualizer (10-11) and the LM visualizer (10-09) are on v15.
- FIG 0 and `make_gate_examples.py` — fixed 2026-09-24: contours and captions use each row's `thr_pos_used` / `thr_neg_used` when the cache has them (schema 3), the gate paragraph states the noise-scaled rule, the bin counts come from the cache (103 × 300 = 30 900, 2 % = 618); regenerate the figures on v11.
- Cut-offs and labels — fixed 2026-09-24: `sweep_by_k.csv` counts "dominant" at ≥ 0.8 like `mixture_summary`; the cNMF manifest now writes `in_sample_variance_explained` (the old key kept for older readers); cNMF run ids are local time from now on.
- Docstrings — fixed 2026-09-24: 240–242 headers say four feature sets; `make_heldout_variance` says all three methods fit in unit-norm; `make_cluster_statistics` / `make_cluster_figures` carry their own header above the copied BSF text; `lf_concat` says 103 × 900.
-3 On 2026-09-22 a cache miss rebuilt `concat_source_v8` in place, so the original v8 cohort (1680) is no longer reproducible from disk — one reason the v8 figures must be regenerated rather than reused.
-4 Which electrode count you quote depends on the tree and the gate: v9 fixed gate 1695; v9 cubes with k = 2 1620; v10 1624; v11 1487{{; v15 1613}}. Read it off the build you are quoting, never off an older line.
