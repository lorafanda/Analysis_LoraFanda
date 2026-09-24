# Read me first
> How to read this tab: every line is a bullet; its darkness is how sure it is. Black = read from the code or from a run on 2026-09-24. Dark grey = the pipeline as I know it (not re-read today). Mid grey = plausible, check before you say it. Light grey = your slot: a citation with its section / paragraph, a number to read off a run, a decision. Edit `02_FBM_Clustering/paper2_outline.md`, then `python make_outline_tab.py --insert` and `python make_site_ui.py --insert`.
> Citation slots are written as chips, e.g. [cite: Leske & Dalal 2019, NeuroImage 189:763, sec 2.3 par 2]. Fill the section / paragraph, or replace the reference.
-4 Journal: Cell Reports (research article: Summary, Highlights, Introduction, Results, Discussion, Limitations, STAR Methods, Figures). Swap the order if the target changes.
- Cohort v10 exists (built 2026-09-24 01:42): 1624 gated / 2961 ungated electrodes, 29 patients (k = 2 gate, EL038 rebuilt, MicroEPI Unknown contacts out, PAT_3415 depth contacts in). Stage-02 numbers below are still v8 / v9 where they say so; k-means and Ward have been refitted on v10, convex NMF, 249, 252 and every figure not yet.

# Summary
-3 Clinical hook: language must be mapped per patient before resection; electrical stimulation mapping is causal but slow, binary and incomplete [cite: Author et al., sec _ par _].
-3 Passive mapping from iEEG scores each electrode active / silent; the question here is what *kinds* of response exist.
-2 Data: 3 naming conditions (auditory definition, picture, written sentence completion), 28–29 patients, sEEG (+ 1 ECoG-only patient excluded), Bern + Geneva + Geneva micro–macro cohorts.
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
- 31 patients processed (5 MicroEPI G-01…G-06 without G-05, 16 Bern EL030–EL052, 10 Geneva PAT); cohort after rules: v9 = 28 patients / 2966 ungated / 1695 gated; v10 = 29 patients / 2961 ungated / 1624 gated (PAT_3415's 24 depth contacts in; PAT_6619 143 → 98 gated under the noise-scaled gate).
- v10 per patient (gated / ungated): EL030 66/83, EL033 74/74, EL034 33/52, EL035 40/98, EL036 27/48, EL037 45/90, EL038 63/66, EL040 35/122, EL042 92/103, EL043 63/67, EL045 90/91, EL046 75/83, EL048 16/59, EL051 29/85, EL052 38/104, PAT_2868 13/39, PAT_3066 33/152, PAT_3390 29/96, PAT_3415 24/24, PAT_3455 51/95, PAT_3780 50/116, PAT_3965 90/169, PAT_3975 79/155, PAT_5515 16/143, PAT_5533 49/164, PAT_6619 98/133, PAT_6704 42/128, PAT_6854 132/175, PAT_6953 132/147.
- Excluded whole patients: EL044 (ECoG only, no depth), PAT_6684 (seizure during the task); PAT_3301 (only picture on disk) contributes nothing.
- Sampling: Bern 1024 Hz (h5 / edf), Geneva 2048 Hz (Micromed TRC), MicroEPI 2048 Hz (mat export with macro + microwire channels).
-2 Gate statistics per patient: 10.8 % (PAT_5515) to 100 % (EL033, EL038, EL043, EL046, PAT_6619) gated in v9; the 100 % patients are the low-trial ones (8–24 trials in a condition) — which is why the gate was changed for v10 (R3).
-4 Demographics table: age, sex, handedness, language, hemisphere of implantation, seizure focus — from the clinical files [your slot].
## R2 · One electrode, one representation (FIG 0A, FIG P.2–P.4)
- Product per electrode × condition: 103 frequency bins (0–398.4 Hz, 3.906 Hz step) × 300 time-normalised bins, dB relative to the pre-stimulus baseline; stimulus = bins 0–149, response period = bins 150–299, the response cue at bin 150.
- Odd / even trial halves stored beside every cube (used for reliability and for the v10 gate noise).
- Example electrodes on the figure: auditory (PAT_3455 OTD7, STG), visual (EL043 IOG7), motor / production (PAT_6953 TPD10), preparatory (PAT_6854 IAG6).
-2 What the dB scale shows: broadband high-frequency increases and low-frequency decreases locked to stimulus and to the cue.
## R3 · The activity gate and the cohort (FIG S0)
- Gate per electrode-condition: ≥ 2 % of bins above +2.2 dB or ≥ 4 % below −3.0 dB; an electrode enters if any condition passes.
- v10 (2026-09-24): a bin counts only beyond max(2.2 dB, 2 × noise) / min(−3 dB, −2 × noise), noise = sd(half1 − half2)/2 of that electrode-condition (the noise of the trial average, ~1/√N: 2.1 dB at 8 trials, 0.9 dB at 45). On the v9 cubes: 1695 → 1620 electrodes; only conditions with < 25 trials are affected (the positive threshold moves only when noise > 1.1 dB).
- v10 cache: 9345 electrode-conditions, noise median 0.92 dB, thresholds raised for 1198 rows (largest ±5.0 dB), 3173 rows pass (v9: 3456 of 9432); halves present for every row.
- Why: with a fixed threshold the gated fraction followed the trial count (Spearman −0.79 across patients); split-half Pearson r of the whole cube was too weak a criterion (r ≥ 0.05 leaves 1446).
-2 Trials lost upstream: 4405 trials in → 3217 used (428 not "correct", 760 by the 1 s post-offset floor, rest > 10 s / IQR). The 1 s floor stays for everyone (decided 2026-09-24).
-3 Report the gate as a coverage statement, not a selection of responders: FIG 0B draws kept vs rejected electrodes on fsaverage.
## R4 · Clustering the responses (FIG 1a–d)
- Same electrodes in every run (v8: 1680; v9: 1695; v10: 1624), four feature sets: `concat_hg` (mean of the 21 bins 70–148 Hz per time bin × 300 × 3 = 900 features), `concat_rawds` (15 bands 1–400 Hz × 30 time bins × 3 = 1350), `concat_bands5` (5 bands 1–20 / 20–70 / 70–170 / 170–270 / 270–400 Hz × 30 × 3 = 450), `concat_bands5z` (bands5 z-scored per band across the whole cohort, units = SD).
- Time downsampling 300 → 30 bins per condition block with scikit-image `resize` (anti-aliased), never across a block boundary.
- Three algorithms: convex NMF (graded loadings), k-means, Ward agglomerative; all fitted in unit-norm space (each electrode's feature vector scaled to unit L2 length so distance measures shape, not amplitude).
- K = 5…30 swept for every (algorithm, feature set); bi-cross-validated held-out variance peaks only for convex NMF (v9: K = 10 / 13 / 13 / 14 for hg / rawds / bands5 / bands5z); k-means and Ward curves rise to the edge of the range (0.36 at K = 8 → 0.48 at K = 30).
-2 At the held-out peaks, 15–45 % of electrodes sit in clusters that are > 50 % one patient (bands5 at K = 14: one cluster = 75 of one patient's 77 electrodes). At K = 8: 0–10 %.
-2 Published cut: K = 8 (largest K with no one-patient cluster on HFA); cluster sizes on concat_hg cNMF K = 8 (v7 run): 248, 321, 159, 236, 216, 203, 126, 210 (largest 18.7 %).
-2 Graded membership: on concat_hg (v9, K = 7 publish fit) 71.6 % of electrodes have no majority component (top weight < 0.5), 1.1 % are dominated (≥ 0.8), median top weight 0.40 — the argument for the graded description.
-4 Decide the paper's K: 8 for every cross-method figure, or the cNMF peak per feature set (10 / 13 / 13 / 14) as 249 uses for the statistics.
-3 Named types (current figure vocabulary): production (rise after the cue in every condition), auditory onset, ramp to cue, suppression (below baseline during the stimulus), reading-only / visual.
-4 Per-cluster numbers to quote at the final K on v10: size, patients per cluster (9–21 on v7), largest patient's share (median 29 %), condition profiles.
## R5 · What survives the analysis choices (FIG 2)
- Correspondence by Hungarian assignment on the correlation of mean-centred loadings (shared-electrode counts when a side is a hard partition); reference solution = convex NMF on concat_bands5; Jaccard of the hard labels per matched pair.
- FIG 2 (built by `00_paper2_figures2_2.py`, K = 8): top = cNMF on the four feature sets, bottom = cNMF / k-means / Ward on concat_hg; mean pairwise ARI at every K 5–30, full ARI and NMI matrices, per-electrode modal agreement after matching (drawn on brains) with a 200-permutation null; per-cluster Jaccard vs reference with its own null; self-stability (Jaccard over 50 refits on 80 % subsamples) and PAC.
-2 Numbers on v8: two algorithms on one feature set agree on ~60 % of electrodes; two feature sets under one algorithm on ~40 % (chance 12 %); 31 % of electrodes are placed identically by all four feature sets (chance 0.4 %); ARI k-means–Ward 0.40, k-means–cNMF 0.34, Ward–cNMF 0.29 on concat_hg at K = 10 (v6 statistics).
- FIG 4 (`00_paper2_figure4_correspondence.py`, K = 8, 1000 permutations): every solution described in the shared concat_bands5z space, centroids matched greedily by Pearson r, overlap as Jaccard and weighted Ruzicka, nulls by plain and within-patient reassignment; FIG 5: Hungarian diagonal share of the contingency table for 9 solutions vs 1000 label shuffles.
-3 The auditory-onset type is recovered by every method; the weakest clusters have no counterpart (Jaccard ≤ 0.1).
-4 All FIG 2 / 4 / 5 files on disk are v6–v8; rebuild on v10 before quoting any number.
## R6 · Where the types sit (FIG 4, LanA)
-2 Language-atlas (LanA, 806-subject probabilistic atlas) proximity predicts more HFA while the prompt is heard and during production, less during visual encoding; |ρ| ≤ 0.16, FDR-cleared because n = 2644 (August contact set).
- Per-type test (`00_paper2_figure3_lana.py`, concat_hg K = 8): Spearman ρ between each cluster's loading and the LanA probability at the electrode; two nulls of 1000 permutations (labels rolled along each shaft; within-patient shuffle), 1000 patient bootstraps for the CI, Benjamini–Hochberg FDR, electrodes with atlas coverage ≥ 0.70 only.
-3 Current draft on the site: 0 of 8 types above the shaft-shift null — recompute on v10.
-4 Decide: is the anatomy result a Results section or a Limitations paragraph?
## R7 · Quality control that shaped the cohort
- Per-patient audit of the ERSP run (141): channels in / neural / Unknown-dropped / aux, bad list, reference route and WM count, notch per condition, trials in / kept, cube counts vs the previous tree.
- Review mode of the LM visualiser: every contact of every patient as a concatenated ERSP tile with its native tissue label, status (data / WM reference / bad / Unknown / not recorded) and split-half r, stripe index, HG.
- Stale-cube sweep (142): files older than the run start are deleted so a rerun cannot leave orphan cubes in the cache (v8 had carried 48).
- Trigger census against the photodiode: every logged trial of EL037, EL040, EL042, EL045, EL046, EL052 sits on a pulse; EL038's tables rebuilt from the full session log (14 audio + 2 reading trials recovered, 2026-09-24).

# Discussion
-3 Response types are a property of what an electrode does, not where it sits (the through-line).
-3 K is a judgement between fit and generalisation: held-out variance cannot see a cluster that is one patient.
-3 Hard partitions of these data are not reproducible across preprocessing; the graded decomposition is the analysis, an argmax map is only shown beside its loadings.
-4 Relation to ESM: a passive map can report a type per electrode, not only active / silent [cite: Author et al., sec _ par _].
-4 Relation to language models of cortex (dual-stream, LanA): [cite: Fedorenko et al.; Hickok & Poeppel — sec _ par _].
-4 What the stimulus-locked vs cue-locked types say about production vs comprehension.

# Limitations
-2 sEEG coverage is patient-specific: a cluster should draw on fewer patients than a random draw; patients per cluster 9–21 vs 25.9 of 27 at random.
-2 Trial counts differ (8–51 per condition after filters); the v10 gate corrects the noise dependence, not the estimate quality.
-2 The time axis is warped (proportions 0 / 0.5 / 0.5): no absolute latencies; the fixation period is not in the cube.
-3 Three Bern patients (EL036, EL037, EL040) carried trial-locked mains stripes in earlier trees; the per-shaft spectrum interpolation was introduced for them — state what the current tree still carries.
-3 Two patients (EL051, EL052) are placed on fsaverage from Lookup MNI coordinates, not from a FreeSurfer recon (≤ ~2 mm MNI152 / MNI305 difference).
-4 Language and site heterogeneity (German / French, Bern / Geneva) [your slot].

# STAR Methods · resources
## Key resources table (software)
- Python 3.11 (Windows); numpy (arrays, FFT, warp); scipy.signal (spectrogram, welch, iirnotch, filtfilt, resample_poly, butter, hilbert, savgol_filter, find_peaks); pandas (+ pyarrow for parquet, openpyxl for the Lookup workbooks); h5py + hdf5plugin (Bern h5, MicroEPI mat v7.3); neo MicromedIO (TRC); mne (read_raw_edf only in stage 01; Brain / surfaces in stage 02); matplotlib (all figures); scikit-learn (k-means, Ward); nibabel (FreeSurfer surfaces, annots, mgz); scipy.spatial cKDTree (vertex lookups); pyvista / VTK (off-screen renders); imageio; Niivue 0.45 / 0.69 (web pages).
- scikit-learn: KMeans (240, cNMF initialisation, consensus, nulls), AgglomerativeClustering (bi-CV), KNeighborsClassifier(1) predictor, PCA(2), silhouette / Calinski–Harabasz / Davies–Bouldin, ARI, NMI; scipy: cluster.hierarchy (Ward in 241), spatial (pdist, cKDTree), optimize (nnls, linear_sum_assignment), stats (spearmanr, t); scikit-image resize; joblib; convex NMF in-house (`functions/lf_decompose.py`).
-4 Versions: not pinned in the repo; read them off the environment before submission (`pip freeze`).
- FreeSurfer recon-all / FastSurfer outputs (elec_recon/*.electrodeNames, *.LEPTOVOX, surf/?h.pial, ?h.sphere.reg, mri/transforms/talairach.xfm); fsaverage; Yeo 2011 7/17-network annots; Desikan-Killiany aparc; Destrieux a2009s for region outlines.
-3 LanA language atlas (Lipkin et al. 2022, 806 subjects) [cite: Lipkin et al. 2022, Sci Data — sec _ par _].
- Code: `lorafanda/Analysis_LoraFanda` (stages 01 preprocessing, 02 clustering / recon); status site and visualisers at `lorafanda.github.io`.
## Resource availability
-4 Lead contact, materials availability, data and code availability statements [your slot].

# STAR Methods · participants and recordings
-2 Patients with drug-resistant focal epilepsy implanted for presurgical evaluation at Inselspital Bern (EL) and Geneva University Hospitals (PAT, MicroEPI); informed consent; ethics approvals [cite: approval numbers].
- Cohorts and files: Bern .h5 (Blosc-compressed from EL052 on) or .edf, 1024 Hz; Geneva Micromed .TRC, 2048 Hz; MicroEPI .mat exports (dataEcog macros, dataMicroDown microwires, photodiode), 2048 Hz; split recordings joined for EL043 and EL051 (RAW_CONCAT).
- Electrode types: sEEG depth contacts throughout; EL044 subdural grids only (excluded); PAT_3415 mixed (64 grid contacts dropped, 57 depth kept); MicroEPI Behnke-Fried style shafts with microwires (microwires removed from the cohort by the lower-case-m rule: `ADm1`, `X1M`…).
-4 Recording hardware, amplifier filters, reference at acquisition [your slot; not in the code].
-4 n patients / n electrodes / n per site in the final cohort [read off v10].

# STAR Methods · task
-2 Language mapping task (paper 1): three blocks (auditory definition naming, picture naming, sentence completion by reading), 53 / 54 / 53 trials, randomised exemplars with a few repeats, spoken responses; PsychoPy log with onset, duration (= response latency from stimulus offset), response_type (correct / wrong), exemplar, epoch timestamp.
- Trial timing from a photodiode on DC6 (Bern) / trigger inputs (Geneva): stimulus onset = pulse rising edge, stimulus offset = falling edge; trial_end = offset + logged latency × fs.
- Photodiode detection (`LFfunctions_PDextract`): 10 Hz 2nd-order Butterworth low-pass (zero-phase), Savitzky–Golay derivative (65.5 ms window, order 5), peaks of the normalised derivative ≥ 0.40 at ≥ 0.2 s spacing; offsets = steepest falling edge before the next onset; per-patient flip for inverted diodes; manual trigger tables for EL030, EL038, EL051, EL052, PAT_3301, PAT_3975, PAT_3780.
- Trial tables: one .tsv per condition in prep0 (`onset, onset_duration, sample, sample_offsets, trial_end, condition_name, resp_accuracy, trial_idx`); byte-identical duplicates skipped by MD5.
-3 Stimulus lengths: pictures 1.0 s, written sentences 3.5 s, audio 3.2–7.2 s (the sound file); identical per exemplar across patients (validated to ≤ 20 ms).
-4 Stimulus material source and language versions (GER / FR) [cite: paper 1, sec _ par _].

# STAR Methods · electrode localisation
- Native: contacts in LEPTOVOX voxel coordinates → surface RAS (tkrRAS) with the mgz vox2ras_tkr (nibabel), k axis flipped; distance to the nearest native pial vertex (cKDTree).
- WM vs cortical: BIDS electrodes.tsv `tissueLabel` first token `wm-` and `tissueWeights_1` > 0.97 → WM (the docstrings say "= 1.0"); Lookup workbook `isWM` for EL051 / EL052.
- To fsaverage: cortical contacts by spherical registration (nearest native pial vertex → subject ?h.sphere.reg → nearest fsaverage sphere vertex → its pial coordinate; snapped to the surface); WM contacts and patients without sphere.reg by affine (tkrRAS → scanner → MNI305 via talairach.xfm → fsaverage).
- EL051 / EL052: Lookup MNI coordinates used as fsaverage RAS (fsaverage surface RAS = MNI305; MNI152 difference ≤ ~2 mm), 100 % matched.
- Labels: Yeo-7 / Yeo-17 at the nearest fsaverage pial vertex; Desikan-Killiany aparc at the nearest vertex (pial in the main cache, white for EL051 / EL052 — a known inconsistency); WM rows labelled "WhiteMatter".
- Name reconciliation between recording and reconstruction (RECON_ALIAS: EL034 MFG→MFG-L, EL043 pl→pIns, EL045 PlanTL→PlaT_L, EL046 al_L/pl_L→aI_L/pI_L, PAT_6619 OFAD/OFPD→OFA/OFP; 68 electrodes recovered) and between recording and BIDS (CHANNEL_SHAFT_ALIAS, 137 contacts in 10 patients).
-4 CT–MRI co-registration and contact localisation software (iELVis? Lead-DBS? in-house) and FreeSurfer / FastSurfer versions [your slot; not in the code].
-3 Subcortical contacts (hippocampus, amygdala) are not `wm-` labelled, so they take the surface route and are snapped to the pial surface — decide whether to keep them at volume position for the anatomy figures.

# STAR Methods · preprocessing (stage 01, `140_ersp_pipeline.py`)
## Channel selection
- Auxiliary channels removed by name (MRK, MKR, X, ECG, EX, AUDIO prefixes; the export stage also drops PHOTO, EKG, EMG, EOG, TRIG, PULSE, RESP, AINP, E1–E8, X1–X8).
- Contacts without a parcellation (tissueLabel empty / "Unknown") dropped in every cohort (since 2026-09-23; 29 MicroEPI contacts).
- Bad contacts from a per-patient manual list (`bad_channels_manual`, visual / clinical review) plus Lookup `isOut`: excluded from every reference mean and given no ERSP cube; still drawn in QC.
- Grid contacts (EL044 P / Pa / T / postP; PAT_3415 GA–GH) and microwires (`…m1`, FOM, X1M) never enter the clustering cohort.
- Bern recordings cropped to the task window (`time_range` per patient); EL043's hemisphere suffix stripped to match its reconstruction.
## Re-referencing
- One reference per patient, computed on the whole cropped recording before notching and epoching, subtracted from every channel.
- White-matter reference: mean of ≥ 3 WM contacts (rule above), bad-listed WM contacts left out; WM reference contacts are not analysed as data (Bern / Geneva) — MicroEPI keeps them as data (selective reference, macros only); `WM_NOT_REFERENCE` keeps PAT_6704 THD1 / THD3 / THD4 as data.
- No usable WM contact: EL052 whole-recording common average (104 channels, 24 bad out); EL044 per-grid CAR (P 6, Pa 55, T 48, postP 15).
-3 [cite: WM referencing in sEEG — Arnulfo et al. 2015, J Neurosci Methods 240:1; Mercier et al. 2017, NeuroImage 147:219 — sec _ par _]
-4 Reference counts per patient: 4–53 WM contacts (audit table) [pick the range to quote].
## Line noise
- Per condition block (trials ± 10 s), on the block's own median Welch PSD (2 s segments, 0.5 Hz): candidate 50 Hz harmonics to 400 Hz (EL048 also a 16.667 Hz railway comb); a harmonic is corrected when its peak z ≥ 3 against the ±1–5 Hz surround.
- Two methods: IIR notch (`scipy.signal.iirnotch`, Q from the peak's sharpness, zero-phase) for most patients; spectrum interpolation (amplitude inside the measured peak width replaced by the flanks' RMS with random phase, width widened up to ±12 Hz until the ring is flat) for EL036, EL037, EL040, EL043, EL045, EL048 and the five MicroEPI patients — decided per shaft for these.
- Audit per harmonic: before / after dB against the floor and residual hole width; unexplained-peak table after the notch.
-2 [cite: spectrum interpolation — Leske & Dalal 2019, NeuroImage 189:763, sec 2.3] ; FieldTrip `dftfilter` with `dftreplace = neighbour` is the same idea.
- No other filtering: no high-pass, no low-pass beyond the anti-alias of the resampling, no detrending.
## Trial selection
- A trial is kept when the response is scored correct, the stimulus lasted ≥ 0.5 s, the post-offset period (offset → trial_end) is 1–10 s, and it is inside the Tukey fences (1.5 × IQR) of the condition's post durations; trials overlapping a listed discharge span are dropped (none in the current cohort).
- Per-trial ERSP rejection exists but is off for every current patient.
- Yield: 3217 of 4405 trials (v9 cohort); per condition 8–51 per patient.
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
- Filters at read: non-neural names, microelectrodes (`is_micro`: lower-case m before the contact number, plus FOM / X1M / X2M; PAT_ only), noisy shafts (PAT_3415 IMG / TA / IPG), grid shafts, excluded patients (EL044, PAT_6684).
- Gate (above): fixed thresholds in v9, noise-scaled (k = 2) in v10; `high_activity` per electrode-condition, an electrode kept if any condition passes; proportions 2 % / 4 %.
- Cohort rule: all three conditions on disk, gated in ≥ 1, one row per electrode; concatenation audio | picture | reading → 103 × 900.
- Cohort sizes: v8 1680 / 27 patients; v9 1695 / 2966 / 28; v10 ≈ 1640 / 29 (PAT_3415 depth contacts in).
-2 Feature sets: `concat_hg` 70–150 Hz mean per bin (3 × 300); `concat_rawds` 15 bands × 30 bins per block; `concat_bands5` 5 bandwidth-weighted bands × 30 bins; `concat_bands5z` the same z-scored per band across the cohort.
-2 Normalisation: each electrode scaled to unit length before fitting (spherical k-means logic; without it added clusters split by amplitude: η² of norm 0.65 → 0.11, ARI 0.22 between the two spaces on concat_hg).
-2 [cite: Hamilton, Edwards & Chang 2018, Curr Biol 28:1860 (per-electrode normalisation before decomposition), Fig S2] [cite: Dhillon & Modha 2001, Machine Learning 42:143 (spherical k-means)]

# STAR Methods · clustering
- Convex NMF (Ding, Li & Jordan 2010, IEEE TPAMI 32:45), in-house NumPy implementation (`lf_decompose.convex_nmf`): X ≈ G (Wᵀ X) with multiplicative updates on the signed Gram matrix; initialised from k-means labels (n_init 10) with G = 0.2 (+1 at the assigned cluster) + 0.2 · U(0,1); stop when the relative Frobenius change < 1e-6 (checked every 10 iterations); W columns normalised to sum 1; loadings G row-normalised (graded membership), argmax only for display.
- Sweep (`sweep_decomposition.py`): K = 5…30, n_iter 1000, random_state 0, on the unit-norm rows of the newest k-means run's X_train; publish fit at K = 7 with n_iter 300 (`run_decomposition.py`).
- k-means: scikit-learn `KMeans(n_clusters=K, n_init=20, random_state=42)` (240); Ward: `scipy.cluster.hierarchy.linkage(method="ward")` on Euclidean `pdist`, cut with `fcluster(maxclust)` (241); both on unit-norm features; silhouette, Calinski–Harabasz and Davies–Bouldin recorded at every K (silhouette's best K is recorded, not used).
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
-3 The site's stage-02 numbers are on v8 (1680 / 27) until 240–242, 249, 252 and the figures are rerun on v10.
- Runs on disk now mix cohorts (2026-09-24 01:49): k-means on concat_hg / rawds and Ward on all four feature sets are v10 (n = 1624); k-means on bands5 / bands5z and every convex NMF run are v9 (n = 1695). `make_cluster_statistics` will refuse until 240–242 are complete on v10; the statistics folders hold v6–v8 (n = 1688 / 1719 / 1680).
- FIG 0 and `make_gate_examples.py` — fixed 2026-09-24: contours and captions use each row's `thr_pos_used` / `thr_neg_used` when the cache has them (schema 3), the gate paragraph states the noise-scaled rule, the bin counts come from the cache (103 × 300 = 30 900, 2 % = 618); regenerate the figures on v10.
- Cut-offs and labels — fixed 2026-09-24: `sweep_by_k.csv` counts "dominant" at ≥ 0.8 like `mixture_summary`; the cNMF manifest now writes `in_sample_variance_explained` (the old key kept for older readers); cNMF run ids are local time from now on.
- Docstrings — fixed 2026-09-24: 240–242 headers say four feature sets; `make_heldout_variance` says all three methods fit in unit-norm; `make_cluster_statistics` / `make_cluster_figures` carry their own header above the copied BSF text; `lf_concat` says 103 × 900.
-3 On 2026-09-22 a cache miss rebuilt `concat_source_v8` in place, so the original v8 cohort (1680) is no longer reproducible from disk — one reason the v8 figures must be regenerated rather than reused.
-4 Which of 1695 / 1620 / ~1640 / 1719 you quote depends on the tree and the gate: v9 fixed gate 1695; v9 cubes with k = 2 1620; + PAT_3415 depth 24; − MicroEPI Unknowns; ± EL038 → read off the v10 build.
