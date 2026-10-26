# Paper 2 · talk of 2026-10-27 · objectives → approach → methods → results

Written 2026-10-26 from the v17 figures, `paper2_outline.md`, `paper2_figure_methods.docx`
(sections 1–11) and the papers on the Desktop. Every number is v17 (1587 electrodes, 29
patients, runs of 2026-10-17) unless marked. Section 4 ends with the decisions that make the
figures say one thing consistently; those are yours.

---

## 1 · Objectives

- **The question.** During a naming task, do intracranial responses fall into a few
  recurring TYPES across patients, and which analysis choices do those types survive?
  "Clustering language" here means clustering the time–frequency response of every electrode
  recorded during three language conditions; it does not mean clustering languages.
- **Why it matters.** Passive mapping scores an electrode active or silent. Electrical
  stimulation mapping is causal but slow, binary and incomplete. A response TYPE per electrode
  is the read-out passive mapping does not give today: what kind of response, not only where.
- **The data.** 29 patients (Bern and Geneva, sEEG; 32 processed, EL031 / EL044 / PAT_6684
  excluded whole), 2917 electrodes with all three conditions, 1587 pass the activity gate.
  Three naming conditions per block: auditory definition, picture, written sentence
  completion; each trial stimulus → cue → spoken answer. 4580 trials in, 3609 used (78.8 %),
  20–51 per condition and patient.
- **What the talk delivers.** (i) The types at the chosen K, with provisional names; (ii) how
  many survive a change of feature description and of algorithm; (iii) where they sit relative
  to the language atlas, with the nulls that say whether that means anything; (iv) the
  patient-generalisation caveat that shaped K.
- **What it does not claim.** Language selectivity (no localiser in the task); absolute
  latencies (the time axis is warped); causal roles; a cluster is what an electrode does,
  not where it sits (the through-line).

---

## 2 · Approach, and what inspired it

### 2.1 · The literature behind each design decision

Screened systematically (ASReview, July 2026): `FBM_Paper2_language` 599 records (1974–2026),
117 screened, 34 included; `FBM_Analysis` 259 records, 162 screened, 51 included. Read closely
(annotated copies on the Desktop):

| Paper | What it gave this paper |
|---|---|
| **Yu, Dugan, Doyle, Devinsky, Friedman & Flinker 2025**, bioRxiv 2024.05.15.594403 — *A left-lateralized dorsolateral prefrontal network for naming* | The closest sibling: 48 patients, NMF on epochs glued end to end (perception, pre-articulation, articulation), five networks. Same move as ours (audio, picture and reading concatenated, so a cluster is a shape ACROSS conditions). Where we differ, and must say why: they use high gamma only (our worst-generalising set), choose K by the elbow of in-sample R², and validate on a second cohort of 23 patients, which we do not have. |
| **Regev, Casto, … Fedorenko 2024**, Nat Hum Behav 8:1924 — *Neural populations in the language network differ in the size of their temporal receptive windows* | k-medoids on high-gamma time courses of 177 language electrodes (6 ECoG patients), k = 3 by an elbow, validated by split-half permutation, electrode-omission robustness and a replication set. Good practice adopted: every cluster shown per participant; split-half reliability as a possible filter before K. Caution kept: an elbow is a visual heuristic with no null; removing random electrodes never removes a patient. |
| **Mercier et al. 2022**, NeuroImage 260:119438 — iEEG guidelines | The field's own statement of our limitation: whether a cortical location can be shown to generalise depends on "whether that particular cortical location is frequently targeted by intracranial surgery" (p. 41). Aggregate on anatomy first (p. 43); grouping electrodes by an INDEPENDENT functional localiser is allowed and not circular (p. 43) — LanA is such a localiser; the smoothing radius is "an empirical question" (p. 41). |
| **Conrad, Bernabei et al. 2020**, Netw Neurosci 4:484 — sensitivity of network statistics to incomplete sampling | Robustness tracks how much was implanted, not only what the brain does; contiguous removal hurts more than random removal (our shaft-wise null); stratify patients by sampling adequacy rather than weight them. |
| **Kochi et al. 2026**, NeuroImage 340:122236 — whole-brain effective connectivity of auditory and visual naming | 127 patients, 9526 electrodes, the same two naming conditions. The operational answer to thin coverage is a FLOOR (ROIs sampled in > 20 patients, > 70 electrodes) and a population-level framing; leave-one-patient-out reported with ρ 0.95–0.99 on ROI aggregates. |
| **Fedorenko, Ivanova & Regev 2024**, Nat Rev Neurosci 25:289 — the language network as a natural kind | What the comparison with LanA can and cannot show: the core network is left frontal + temporal, interconnected, modality-independent and language-SELECTIVE, i.e. distinct from speech perception, articulation and the visual word form area — the very systems a naming task drives. |
| **Lipkin et al. 2022**, Sci Data — LanA (806 subjects) | The probabilistic atlas sampled at every electrode (1587 of 1587 covered on the current coordinates). |
| **Ding, Li & Jordan 2010**, IEEE TPAMI 32:45 — convex and semi-NMF | Convex NMF: centroids are convex combinations of the data, loadings are non-negative and graded, so an electrode can belong partly to two types. |
| **Hamilton, Edwards & Chang 2018**, Curr Biol 28:1860 — onset and sustained responses in STG | Per-electrode normalisation before decomposition; a data-driven two-type map as the precedent for "types" in iEEG. |
| **Hamberger 2007**, Neuropsychol Rev — cortical language mapping in epilepsy | The clinical hook: what ESM gives and where it falls short. |
| Also on the desk | Owen & Perry 2009 (bi-cross-validation), Hennig 2007 / Ben-Hur 2002 / Lange 2004 (stability), Monti 2003 (consensus), Alexander-Bloch 2018 (spin test), Hubert & Arabie 1985 (ARI), Miller 2007 (spectral changes with movement), Hickok & Poeppel (dual stream), Fanda et al. (paper 1, the paradigm and the HFA map). |

### 2.2 · So the design is

1. **One representation per electrode, all conditions at once.** A time-normalised ERSP cube
   per electrode × condition, concatenated audio, picture, reading (after Yu 2025), so a type
   is a shape across the whole task, not a response in one condition.
2. **Several frequency descriptions, not high gamma alone.** 15 bands, 5 bands, 2 bands, HFA.
   The HFA-only description generalises worst across patients (result of 2026-10-14), and it
   is the one both siblings used.
3. **Graded membership.** Convex NMF gives every electrode a loading on every type; the argmax
   is used for display and counting only. k-means and Ward run beside it as hard-partition
   controls.
4. **K by patient generalisation, not by an elbow.** Held-out variance (bi-cross-validation)
   never turns over on these data; the published K is the largest at which no cluster is more
   than half one patient (the Regev / Yu caution; Conrad's and Mercier's point about sampling).
5. **Validation by agreement and nulls, since there is no second cohort.** Agreement across
   feature sets and algorithms (FIG 2), correspondence of individual clusters against random
   pairings (FIG 4), every solution against every other (FIG 5), stability under refits.
6. **Anatomy as context, with coverage-aware nulls.** Electrodes on fsaverage; LanA as an
   independent localiser (Mercier's rule); a shaft-shift null that keeps spatial smoothness,
   a within-patient null, and a spin null that randomises the map instead of the electrodes.

---

## 3 · Methods (condensed; the full version is `paper2_figure_methods.docx`, sections 1–11)

1. **Recordings.** Bern 1024 Hz (h5 / edf), Geneva 2048 Hz (Micromed TRC), MicroEPI 2048 Hz
   (mat, macro + microwire). Electrodes localised per patient and carried to fsaverage
   (cortical contacts by the subject's spherical registration, white-matter contacts by the
   Talairach affine).
2. **Preprocessing (`140_ersp_pipeline.py`).** Channels: auxiliary and unlabelled contacts
   removed, a manual bad list per patient. Reference: the mean of the patient's white-matter
   contacts (tissue weight > 0.97, at least 3), not analysed as data. Line noise per condition
   block: 50 Hz harmonics to 400 Hz plus a patient's own comb where found, corrected by spectrum
   interpolation when the peak is z ≥ 3 against its surround. Trials: correct answer, stimulus
   ≥ 0.5 s, post-offset period 0.2–10 s inside 1.5 × IQR fences, per-channel HG outlier
   rejection. Baseline −0.4…−0.1 s.
3. **The ERSP cube.** 103 frequency bins (0–398 Hz, 3.9 Hz step) × 300 time-normalised bins,
   dB re baseline; stimulus = bins 0–149, cue at 150, response period 150–299. Odd / even trial
   halves stored beside every cube.
4. **The activity gate.** An electrode-condition passes with ≥ 2 % of bins above +2.2 dB or
   ≥ 4 % below −3.0 dB, thresholds raised to 2 × the noise of the trial average where that is
   larger (noise = sd(half1 − half2)/2; median 0.90 dB; raised on 559 of 8751 rows). An
   electrode enters if any condition passes: 2935 rows pass, 1587 electrodes.
5. **Feature sets.** Time 300 → 30 bins per block (anti-aliased resize, never across a block
   boundary); bands 15 (1–400 Hz) / 5 / 2 (30–50, 70–150 Hz) / HFA (70–150 Hz) × 30 × 3 =
   1350 / 450 / 180 / 90 features; every electrode's vector scaled to unit length, so distance
   measures shape, not amplitude.
6. **Clustering.** Convex NMF (in-house, initialised from k-means, 1000 iterations, K 5–30);
   k-means (20 restarts); Ward. Twelve solutions = 3 algorithms × 4 feature sets, all on the
   same 1587 electrodes.
7. **Choosing K.** Bi-cross-validation (4 × 4 folds, 300 iterations): the held-out curves rise
   to the edge of the range (peaks K 30 / 30 / 29 / 29); at those K, 54 / 58 / 25 / 34 % of the
   electrodes sit in clusters that are more than half one patient. Published K = the largest
   with no such cluster: 7 / 8 / 10 / 6 (15 bands / 5 bands / 2 bands / HFA). Cross-method
   figures at K 7 on 15 bands.
8. **Stability.** 50 refits on 80 % subsamples per run (Jaccard, PAC) = the ceiling in FIG 2;
   patient-resampled consensus with a covariance-matched Gaussian null written
   (`consensus_stability.py`), not yet run.
9. **Agreement and correspondence.** ARI / NMI between partitions, per-electrode modal
   agreement after Hungarian matching against 1000 permutations (FIG 2); every cluster
   described in the shared 5-band space, matched by Pearson r, overlap as Jaccard against
   5000 random pairings (FIG 4); Hungarian diagonal share over all 66 pairs of solutions,
   1000 label shuffles (FIG 5).
10. **Atlas.** LanA sampled at each electrode's fsaverage coordinate (nearest 2 mm MNI152 voxel).
    Per cluster: mean P(LanA) and the share inside the network at P ≥ 0.1 and ≥ 0.2. Nulls:
    5000 shaft-shift permutations (labels rolled along each shaft, atlas fixed), 5000
    within-patient shuffles, 2000 patient bootstraps, Benjamini–Hochberg over K. FIG 3b: 5000
    spins of the map on the fsaverage sphere, plus the network's own response inside vs
    outside.
11. **Per-cluster statistics (249).** Silhouette against 50 Gaussian surrogates with the data's
    covariance; patients per cluster against 400 size-matched random draws; leave-one-patient-
    out placement.

---

## 4 · Results — what each figure shows now, its one sentence, and what to decide

### 4.1 · Figure by figure (15 bands, K 7, cNMF unless stated)

- **FIG 0 / S0 — the task and the cohort.** Four example electrodes (auditory, visual, motor,
  preparatory) on the ±7 dB scale; every electrode the gate saw on fsaverage, kept vs rejected;
  per-patient composition, atlas coverage. *Sentence:* 1587 of 2917 electrodes respond, in
  29 patients, with a gate that scales to each electrode's own noise.
- **FIG 1a–d — the types.** A held-out variance vs K; B one block per cluster (mean ± SD over
  the three conditions, hemisphere renders, patient composition); D the one-patient-cluster
  curve. Sizes 318 / 143 / 168 / 200 / 334 / 200 / 224; patients per cluster 11–28 (median 22;
  random draw 27–29); largest patient's share 12–36 % (median 20 %; random 9 %, worst c2 36 %
  PAT_3965). Provisional readings: c5 production (HFA burst after the cue, every condition),
  c0 beta decrease (8–30 Hz down through the trial), c2 broadband decrease (70–400 Hz down
  during stimulus and response), c4 low-frequency rise (1–8 Hz up during the stimulus); c1
  audio-stimulus low-frequency, c3 reading-stimulus low-frequency, c6 response-period
  low-frequency fall. *Sentence:* seven recurring shapes, none owned by one patient — at the
  held-out peak, half the electrodes would be.
- **FIG 2 — what survives the analysis choices.** ARI across feature sets 0.07–0.38 (mean
  0.15), across algorithms 0.26–0.51 (mean 0.37); 11.4 % of electrodes placed alike by all four
  sets (chance 0.6 %), 47.0 % by all three algorithms (chance 3.1 %); refit ceilings cNMF 15
  bands 0.82, k-means 0.76, Ward 0.64; 1000 permutations. *Sentence:* the algorithm matters
  less than the frequency description.
- **FIG 4 — the same clusters more often than chance.** Diagonal share 52–73 % between
  algorithms vs 14 % random; 21–60 % between feature sets (HFA lowest); 36 of 42 matched pairs
  clear both nulls; c2 found by every algorithm (Jaccard 0.83 k-means, 0.64 Ward), c1 missed by
  both, c3 and c5 by Ward. *Sentence:* the broadband decrease and the production type are the
  robust ones; the low-frequency types are the fragile ones.
- **FIG 5 — twelve solutions against each other.** 66 pairs, median diagonal share 0.55 (same
  set, different algorithm) / 0.38 / 0.36, chance 0.18, no pair at chance; K by K, cNMF on 15
  bands keeps 0.90 of its partition from K 7 to 8. *Sentence:* the partition is stable in K and
  far from chance everywhere.
- **FIG 3 — where the types sit (LanA).** New panel A: violins of P(LanA) per cluster, every
  electrode a point greyed by its loading (second file: hollow below 0.3). Cohort mean 0.142;
  c1 0.207 beats only the within-patient null (q 0.001; shaft-shift q 0.37); c1 and c5 keep 55 %
  of their electrodes inside the network at P ≥ 0.1 vs 41 % in the cohort, c0 and c6 32–33 %;
  nothing beats the shaft-shift null. FIG 3b (spin null, dry run): no cluster beyond a rotated
  map; the network's own response is not distinguishable from the rest of the sampled cortex
  (p 0.34 all electrodes, 0.12 left). *Sentence:* the types are not the language network, and
  with a naming task that is expected — the atlas subtracts exactly the perception and
  articulation systems the task drives. (Decide: a results slide or a limitation line.)
- **Generalisation.** At the chosen K no cluster is one patient; leave-one-patient-out
  placement tracks electrode count (ρ −0.52) more than shared coverage (ρ +0.36, p 0.055):
  holding a patient out removes REGIONS, not a random sample. *Sentence:* what fails to
  generalise may be what only two patients had electrodes near.

### 4.2 · Decisions that make the figures consistent (your call; the current state in the middle column)

| Parameter | Now | Proposal |
|---|---|---|
| K | 7 / 8 / 10 / 6 by the dominance rule; cross-method figures at K 7 | Keep. Show the held-out peaks only as the triangle on FIG 1A. |
| Membership used in every count and test | argmax of the sum-normalised loadings (FIG 1 sizes, FIG 2–5, FIG 3 tests) | Keep; the loading changes how an electrode is DRAWN, never a number. Say so once. |
| Loading display rule | FIG 1 and the visualizer: grey at 1/K, full colour at 0.5, size to 0.7, opacity floor 0.1. FIG 3 A (today): grey at 0.2, full at 0.6; the hollow file cuts at 0.3 | One rule everywhere electrodes are drawn: either FIG 1's (full colour at 0.5) or the 0.3 cut. Not both thresholds in one talk. If 0.3 is the rule, FIG 3 A keeps the hollow file and FIG 1's renders need the same cut (one constant, COLOR_FULL). |
| Which loading file | the figures read the K-sweep loadings (1000 iterations: own loading median 0.41, range 0.20–0.88, 15.5 % below 0.3); the outline's "median top loading 0.48, 27 % ≥ 0.5" was read from another fit | Quote the sweep numbers and fix the outline line. |
| LanA thresholds | 0.1 and 0.2 | Show 0.1 in the talk, keep 0.2 in the table. |
| Null sizes | FIG 2 1000 permutations; FIG 3 5000 + 5000, 2000 bootstraps; FIG 3b 5000 spins (to run); FIG 4 5000 at K 7 / 10, 1000 at K 8; FIG 5 1000 | Rerun FIG 4 K 8 at 5000 so one sentence covers FIG 4. |
| Stars | `*` shaft-shift, `(*)` within-patient | Keep; say "spatial null" and "within-patient null" aloud. |
| Type names | provisional readings of the centroids (mine, not yours) | Name them before the talk; the graphical abstract carries them. |
| Hemisphere | LanA is left-lateralised; 715 left vs 872 right electrodes | One line whenever FIG 3 is shown; FIG 3b has the left-only strip. |
| Time axis | warped 0 / 0.5 / 0.5, cue at bin 150, baseline −0.4…−0.1 s, ±7 dB cube scale | State once on the first ERSP slide. |
| What enters from 249 | silhouette vs the Gaussian cloud, LOPO, patient composition | The composition bar (FIG 1 B) and FIG 1 D only. |

Still to run before the slides: your redraw of FIG 3 (`--n-perm 5000 --n-boot 2000`, screen
unlocked), FIG 3b at 5000 spins, FIG 4 K 8 at 5000 if you take that row.

---

## 5 · A slide order that follows the four sections

1. Title. 2. The clinical hook and the gap (ESM vs passive maps; where, not what). 3. The
objective in one sentence. 4. Task and cohort (FIG 0). 5. The six design decisions and the
papers behind them (the table of 2.1, shortened). 6. From raw signal to cube to gate (FIG S0,
P.2–P.4). 7. Features, convex NMF, choosing K (FIG 1 A + D). 8. The seven types (FIG 1a B).
9. What survives (FIG 2; one panel of FIG 4 or FIG 5). 10. Where they sit (FIG 3 A + B; FIG 3b
as backup). 11. Limitations (coverage, leave-one-patient-out, warped time, references).
12. Next (v18 cache with the blob gate, consensus stability, the names).
