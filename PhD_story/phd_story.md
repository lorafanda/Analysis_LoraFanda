# PhD Project Story — Functional Brain Mapping via Full-Spectrum ERSP Analysis

---

## The Story

### The Problem

Epilepsy patients undergoing resective surgery require presurgical functional brain mapping to identify eloquent cortex before resection. The clinical gold standard — electrical stimulation mapping (ESM) — is invasive, time-consuming, sequential, and limited in coverage. It cannot survey all electrodes simultaneously, is subject to patient fatigue, after-discharge risk, and tests only a fraction of the 100+ implanted electrodes. It is also a binary output: active or not active.

Functional brain mapping (FBM) through passive recording of task-related neural activity offers a complementary approach. If a patient completes a 40-minute FBM battery covering auditory, visual, reading, and motor conditions, the resulting intracranial EEG data contains rich time-frequency information across every implanted electrode simultaneously. The problem is that this data is currently either not used systematically to guide ESM, or when it is used, it is filtered to a predefined bandwidth — typically high-gamma (70–150 Hz) — before analysis. This is a top-down assumption about which frequencies matter, imposed before looking at the data.

The literature itself argues against this assumption. Crone et al. (1998) established that multiple frequency bands carry distinct and simultaneous functional information — gamma increases and alpha/beta suppression coexist during the same task period and together tell a richer story than either band alone. Therefore filtering to a single preset band before analysis discards known signal. Event-related spectral perturbation (ERSP) preserves the full time-frequency representation. That is the methodological starting point.

---

### Part 1 — The Clinical Contribution: Activity Cards

**What we propose:** a data-driven, full-spectrum pipeline for electrode activity characterization that does not assume which frequency band is relevant.

For each electrode that shows any significant time-frequency departure from baseline in at least one condition, the pipeline generates an **activity card** — averaged ERSPs and high-gamma trial plots across *all* conditions, regardless of which conditions passed the threshold. The cross-condition view within a single electrode is the key: an electrode that is active during response-required language tasks but not during passive image viewing, and also shows strong motor mouth movement activation, reveals its functional identity through the *pattern across conditions*, not through a single binary label.

The segmentation step — blob detection on the full ERSP map — finds these departures without presupposing their frequency location. Simultaneous gamma increase and alpha suppression at the same time point are both captured and both contribute to the electrode's characterization. This is not possible with a preset bandwidth filter.

The output is a ranked set of candidate electrodes that could guide ESM prioritization — telling the clinical team which electrodes are most likely to be functionally relevant before a single stimulation is applied.

**What this is not:** a replacement for ESM. ESM remains the ground truth for critical site identification. This is a tool to make ESM more efficient and more informed.

---

### Part 2 — The Scientific Contribution: Cross-Patient Clustering

**What we propose:** having characterized ERSP response shapes across all electrodes and all patients as geometric blob descriptors, we ask whether unsupervised clustering can recover functionally and anatomically coherent response motifs that are reproducible across patients, recording centers, and languages.

The blob descriptor captures the shape of a response in time-frequency space — its onset, peak frequency, temporal and spectral spread, covariance, mean amplitude, and area — without assuming which frequency band it lives in. These descriptors are extracted from variable-duration trials using a time-normalization warping approach that aligns trials by their functional structure (baseline / stimulus / post-stimulus) rather than forcing a fixed time window.

Clustering these descriptors across patients produces **cluster cards** — each cluster summarized by:
- Task condition distribution (which conditions contributed to this cluster)
- Patient distribution (how many patients and which ones)
- Recording center (Geneva HUG, Bern, MicroEPI)
- Language (French, English, German speaking patients)
- Parcellated brain region (from FreeSurfer anatomical parcellations)

The cluster card does not label a cluster as "planning activity." It reports convergent evidence: if a cluster is predominantly composed of response-locked language trials, from perisylvian electrodes, across multiple patients, centers, and languages, that convergence is itself meaningful. The interpretation is made by the clinician and neuroscientist reading the card, not by the algorithm.

**The falsifiable hypothesis:** electrodes assigned to the same cluster should show higher ESM concordance with each other than electrodes assigned to different clusters. ESM data from the same patients is available to test this, acknowledging that ESM coverage is partial (typically a subset of the 100+ implanted electrodes).

Clusters will be validated by clinical experts.

---

### The Thread Connecting Both Parts

The brain is individual, variable, and reorganized by pathology. Any method that assumes a fixed functional bandwidth imposes a group-level prior onto a signal that needs to be read at the individual electrode level. The pipeline reads the full ERSP, finds what is actually there, characterizes it geometrically, and then asks — across individuals — whether those characterizations group into something reproducible and interpretable.

Part 1 serves the individual patient. Part 2 serves the field.

---

### What Makes This Methodologically Novel

The combination of:
1. Full-spectrum ERSP without preset bandwidth filtering
2. Time-normalization warping for variable-duration trials
3. Geometric blob descriptors that capture response shape without frequency assumptions
4. Cross-patient unsupervised clustering of those descriptors
5. Multi-dimensional cluster cards validated against ESM and anatomy

None of these five elements in combination exist in the published literature. Each individual element has precedents. The combination, applied to heterogeneous multi-center multi-language iEEG data, does not.

---

### Honest Limitations

- Blob segmentation thresholds (`thr_pos`, `thr_neg`, `alpha`) are heuristic and not statistically derived from a null distribution. This needs to be addressed.
- Blob ordering instability (the correspondence problem) is a known vulnerability in the feature vector construction.
- Clusters are currently unvalidated. The clinical guidance claim depends entirely on that validation.
- ESM coverage is partial — validation against ESM will have gaps.
- The pipeline infrastructure handles heterogeneous data well but cross-patient ERSP comparability depends on tight normalization that has not been formally verified across recording systems.

---
---

## Working Notes — Raw Q&A That Built This Story

*The following is the verbatim conversation used to construct the narrative above.*

---

**Q: What is the clinical or scientific problem you are trying to solve — in one or two sentences, as you would explain it to a non-specialist?**

A: In neuroscience, we know from previous studies that the brain is unique from person to person and one person's functional location for example word processing is different from one to another, for various factors. This is especially heavier when looking at it from the point of view of an ill brain ie epilepsy patients, who could have a tumor, or any other morphological chemical differences in their brains. For this reason, it is difficult to say that this region is definitely for audio motor etc etc. For epilepsy patients in particular, it is important to know which part of the brain is a specialisation for which kind of functioning, as before resection operations the medical team sometimes performs stimulation mapping to understand what part of the brain is doing what function. The issue is that this gold standard approach has its limitations, of course. it takes a long time to look at all electrode pairs across the brain, and patients can get tired and it needs to be stopped and whatever other reason. Plus it includes stimulation, which isn't a passive approach. This is where FBM comes in. In order to help guide ESM we build these experiments to have an idea of what is happening across all electrodes at the same time that the patients is perceiving and responding to stimuli of various kinds (audio visual reading motor movements etc etc). So, in short, if we use FBM to analyse ERSPs of brain activity and find electrodes that are responding highly to that stimulation, we could use that information to guide electrical stimulation mapping for a more precise approach.

Additionally, we know that different modalities of brain activity, whether that is motor listening, processing, planning, responding etc etc have different wavelengths of activity difference from baseline. We want to inspect that more with the ERSPs. Can we use methods of unsupervised learning to group these kind of frequency over time patterns? What's the best way to do it? What kind of information are we getting out of it? Can we redefine the frequency ranges of importance for each condition (audio, reading, motor, visual etc etc) in a cross-patient way? This part is a lot more exploratory.

---

**Q: You described FBM as a way to guide ESM rather than replace it. Is that the exact position you want to take — complementary tool — or do you want to push harder and say FBM could reduce the need for ESM in some cases?**

A: No not in this case.

---

**Q: You mentioned epilepsy patients specifically, but your pipeline handles PAT_, EL_, and MicroEPI families across different recording systems and sites. Is the multi-site, multi-system, heterogeneous patient aspect a problem you are explicitly trying to solve — or is the heterogeneity just a practical reality of your dataset that you had to handle but is not itself a scientific claim?**

A: Yes. I think a good pipeline is a pipeline that is across centers, languages, modalities of data collection etc. That's a plus but not something that we advertise.

---

**Q: You described two parts to your project — the clinical FBM/ESM guidance part, and the more exploratory unsupervised clustering part. Are these two separate stories that happen to share the same pipeline, or is the clustering part the method by which you achieve the clinical goal?**

A: I would love to do that. I would love to find out that during clustering we managed to find a cluster that was particular for planning activity pre response and it was across patients across centers and very localized meaning electrode locations were close by. All this done by the frequency over time plots.

---

**Q: When you say "functionally interpretable" — is the interpretation coming from the cluster itself or do you need an external label to validate it?**

A: That's the difficult part. I can't label it because I can't be sure. The way things are generally done is by looking at the ERSP shape and cross referencing with type of trial it was (ie if it's a passively looking at images trial or is it a trial that waits for a response like the language mapping trials).

---

**Q: Is the level of claim you are comfortable making that the cluster appears predominantly in response-locked language trials, in electrodes located in region X, and has this spectral shape — or are you hoping the anatomy does the heavy lifting for interpretation?**

A: I wouldn't say heavy lifting but I would definitely say that the location is considered in making that claim. In my mind this is what the cluster cards are going to accomplish. Meaning, for each cluster, there will be a clear statistical section explaining how many samples of that cluster belong to which (i) task condition (ii) patient (iii) data collection center (iv) language (some patients are French, English, or German speaking) and (v) annotated brain region (from parcellations).

---

**Q: Is the gap that FBM produces rich time-frequency data across all electrodes simultaneously but currently that data is either discarded after guiding ESM or summarized too crudely — and your contribution is a method to extract, group, and characterize those patterns in a principled way across patients, centers, and languages?**

A: Well no the data isn't even used to guide ESM — that's what I'm proposing as part 1. My proposal output for this part is to extract electrodes of high response meaning that had a change in brain activity from baseline and create a cluster card for each electrode. The way that it is currently done in literature is often filtering the signal to the gold standard defined bandwidth but I think that is limiting.

Part 2 is actually more exploratory. We know that brain activity for more simple tasks can be localized in the way that it presents itself. For example, visual cortex is more in the occipital, auditory in temporal, but complex language comprehension and word finding tasks are across the brain and the brain activity changes over time as different parts of the network are activated in propagation.

---

**Q: So part 1 is: instead of filtering to a predefined bandwidth, you let the ERSP speak for itself — you find electrodes that show any significant departure from baseline across the full time-frequency space, characterize that departure as a blob, and produce an activity card per electrode. Is that accurate, and does part 2 summary also hold?**

A: Yes but at this point it won't be a cluster card, it would be an activity card. Meaning, for that electrode which has passed the threshold and segmentation part for at least one condition, we will display the averaged ERSPs and the HG plots for all conditions even if only one condition has passed the threshold. This way we can say something like: the conditions where response was required are active, but not when passively looking at an image, and also the motor movement for mouth movement is highly active — this is likely a motor electrode.

The part 2 makes sense for as much as I understand it.

---

**Q: Is the scientific claim you want to make that predefined bandwidth filtering misses clinically relevant responses — or is it more modest, that your approach is simply more systematic and data-driven?**

A: Yes and I have proof of this, but I can add this section later in my explanation, but trust me the ERSP captures even differences between bands at exactly the same time period. Meaning, there are times when gamma band is active but alpha is suppressed for certain tasks, compared to baseline.

---

**Q: When you say "tell you which cluster each electrode fits into" — can you explain the full pipeline vision for a new patient?**

A: The clusters will be validated by clinicians. I do have ESM data on the same patients that I will test it against for sure although there are a lot of holes in it. In general, ESM is performed in a handful of channels, not all of them (100+). The real claim is: for a new patient, the pipeline outputs not just "this electrode is active" but "this electrode's response pattern resembles cluster X, which across N previous patients was predominantly found in region Y during task Z." That is genuinely beyond what ESM tells you because ESM is binary and sequential. The system is probabilistic and simultaneous.

---

**Q (critic): On "the literature only uses HGA" — oversimplification?**

A (response): Ok it just solidifies why using multiple bands is important, confirming our choice to go with ERSP rather than time series of a filtered preset band. The literature established that multiple bands carry distinct and simultaneous functional information — therefore filtering to a single preset band before analysis discards known signal. ERSP preserves all of it. That is why we use ERSP.

---
