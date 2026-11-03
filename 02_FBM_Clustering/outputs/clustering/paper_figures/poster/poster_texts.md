# Poster texts and captions (cohort v17, 1587 electrodes, 29 patients; drafted 2026-11-03 from the validation story and the current files)

Every number below is the published one on cohort v17 (numbers.json, heldout.json, the MMD run of 2026-10-30, the consensus sweep of 2026-10-26). Box texts are 40-60 words; the title of each box is the claim.

## Title, two options
- Seven response types in intracranial recordings during language tasks, and how far they can be trusted
- Functional brain mapping without stimulation: recurring spectral response types across 29 patients

## Row 1

**Why.** Functional brain mapping before epilepsy surgery relies on electrical stimulation, which is slow and subjective. Task-evoked spectral responses could map function without stimulation, if their patterns recur across patients. We ask whether the responses of 1587 electrodes in 29 patients fall into recurring types, and how far those types survive a change of description, of algorithm and of patients.

**The task and the cohort.** Three naming tasks: auditory definition, picture naming and sentence completion, each a stimulus period and a spoken response. Depth electrodes in 29 patients (Geneva and Bern): 2917 contacts seen, 1587 kept by a response gate (at least 2 % of time-frequency bins above +2.2 dB); the gate keeps 6 % to 95 % of a patient's contacts.

**The pipeline.** Each electrode's event-related spectral perturbation, time-normalised, is summarised as 15 frequency bands x 30 time bins x 3 conditions (1350 values in dB), scaled to unit length and decomposed by convex NMF, which writes every electrode as a mixture of K types. K is the largest number at which no cluster is more than half one patient: 7.

## Row 2, the spine

**Seven response types.** (strip caption) The seven types at K 7: the mean ERSP of each type's members (+-2 dB, audio | picture | sentences) and its electrodes on the glass brain (loading >= 0.3 drawn; colour, size and opacity follow the loading). Ordered by how alike the three conditions look. c5 production n 200, 17 patients; c0 beta decrease 318, 22; c2 broadband decrease 168, 11; c6 response-period low-frequency fall 224, 24; c4 low-frequency rise 334, 28; c1 audio-stimulus low-frequency 143, 22; c3 sentence-stimulus low-frequency 200, 25. Median loading on the own type 0.41: the types are regions of a continuum, not islands.

**The z-scored companion strip.** (caption) The same decomposition with every band z-scored across the cohort, so that high gamma counts as much as the low frequencies; its clusters are ordered and coloured by the published type they are matched to (Jaccard in the header). An auditory high-gamma type appears (v0, right superior temporal, 120 electrodes, 20 patients), and the one-patient rule is already broken at K 6.

## Row 3, the three questions

**Agreement: other descriptions and other algorithms find the same shapes.** k-means and Ward on the same features recover the production, beta-decrease and broadband-decrease types (ARI 0.51 and 0.26 against convex NMF); coarser descriptions agree less (ARI 0.38 with 5 bands, 0.07 with HFA only). Across algorithms every matched pair clears both the plain and the within-patient null; across feature sets fewer do, down to 2 of 7 against HFA. Agreement is not correctness, so it is one test of three.
Caption (Fig 2C): matched clusters, convex NMF against k-means and Ward, 15 bands, K 7: colour is the centroid correlation, the box says whether the pair clears both nulls (5000 draws).

**Stability: the shapes come back when patients are resampled.** Refitting on 80 % of the patients 200 times returns the same partition: the proportion of ambiguous electrode pairs is 0.23 against 0.43 +- 0.03 for a Gaussian cloud with the data's covariance. Per type the consensus runs from 0.81 (low-frequency rise) to 0.53 (sentence-stimulus). At K 7 the largest share of a cluster held by one patient is 36 %; at K 8 one cluster is a single patient.
Caption (Fig 3 B-D): the consensus matrix at K 7, the per-type consensus (mean over the pairs inside a type), and the proportion of ambiguous pairs against the Gaussian cloud over K.

**Separation: an unseen patient's electrode lands in its type nine times in ten.** A model refitted without a patient places that patient's electrodes in their type 90 % of the time (chance 52 %); refitted on half the patients, 62 %. The two low-frequency stimulus types leak most (78 % and 80 %). Seven of 21 type pairs sit closer than two sectors of a featureless cloud, so the types are neighbours in a continuum, distinct but not islands.
Caption (held-out confusion): where each type's electrodes land when their patient is left out, as % of the type's held-out electrodes; the diagonal is the recall.

## Row 4

**Where each type stands.** Each type on all three questions: overlap with the other algorithms and feature sets, consensus, held-out recall, silhouette against the cloud. Broadband decrease and low-frequency rise are solid on every count; production is solid under patients but weak across methods; the audio-stimulus and sentence-stimulus types are the fragile ones. A cell's colour is its rank in the column, not a score.
Caption (Fig 5): the scorecard; every column is one measure, every row one type.

**Where they sit.** Against the probabilistic language atlas (LanA) no type is placed by the network beyond one modest effect: the production type's share inside the network clears a shaft-shift null (q 0.04) and the audio-stimulus type's left-hemisphere mean beats 5000 spun maps (q 0.03). Mean P(LanA) runs from 0.11 to 0.21 against 0.14 for all electrodes. The types are task phases, not the atlas.
Caption (Fig 6A): P(LanA) per electrode, by type, against the cohort's own distribution.

## Row 5

**Conclusions.**
- Seven recurring response types, each carried by 11 to 28 patients, defined by task phase and frequency band.
- The frequency description changes the partition more than the algorithm does.
- The partition is stable under patient resampling and separable on unseen patients, but only partly: two low-frequency types leak into their neighbours.
- Equalising the bands buys an auditory high-gamma type and loses the one-patient rule two steps earlier: the next question.

**Acknowledgements.** We thank the patients who agreed to participate in these studies, and the medical, technology and nursing staff of the epilepsy monitoring units of Geneva University Hospitals and Inselspital Bern for their cooperation. Supported by Swiss National Science Foundation grant 194507.

## Figure files
- Spine: outputs/clustering/paper_figures/poster/concat_bands15_strip.png (the seven blocks; block_<pos>_c<id>.png singly)
- Companion: outputs/clustering/paper_figures/poster/concat_bands15z_strip.png
- Agreement: outputs/clustering/paper_figures/paper/FIG4_K07_run20261017-194301.png, panels A and C
- Stability: outputs/clustering/consensus/concat_source_v17/consensus_bands15_cnmf_K7_slide.png
- Separation: outputs/clustering/paper2_validation/walkthrough/panels/p3_confusions.png, matrix A
- Scorecard: outputs/clustering/paper2_validation/figures/fig5_scorecard.png
- Atlas: outputs/clustering/paper_figures/paper/FIG3_lana_concat_bands15_K7_run20261017-194301.png, panel A
- Task and cohort: the new opening figure once redrawn (00_paper2_figure0_coverage.py); until then paper/FIG0_cohort_v17.png
