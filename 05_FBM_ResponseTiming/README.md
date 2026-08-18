# 05_FBM_ResponseTiming

**Do the functional groups we already found differ in *when* they respond?**

Everything upstream of this folder is time-normalised: 140 warps every trial to 300 bins,
so 0% is stimulus onset and 50% is the GO cue, and a latency can only ever be expressed as
a percentage of a trial whose real duration was thrown away. The clustering (02) and the
pooling roles (04) are both built on that warped data. This folder asks the question the
warp cannot answer: **in real seconds after the GO cue, in what order do those groups
respond, and does the order hold across conditions?**

## Input

`01_FBM_Analysis/outputs/05_ERSP_LM_RAWONLY_RealTime/<pid>/LM/ERSP_matrix/<cond>/`
`<pid>_<cond>_<ref>_ERSP_<chan>_RT_GO.npy`, shape `(129, T)`, written by
`150_ERSP_analysis_pipeline_noTwarping.ipynb` with `RT_ALIGN="go"`,
`RT_WINDOW=(-2.5, 5.0)`, `RT_MAX_POST_S=5.0`.

**Neither axis is stored in the .npy.** They are reconstructed in `functions/lf_rt.py`,
from the same expressions the pipeline used, and that is the only place this knowledge
lives:

```
x = np.linspace(-2.5, 5.0, T, endpoint=False)     # t = 0 is the GO cue
f = np.linspace(0, 500, 129)                       # nfft=256 at fs_ds ~ 1 kHz
```

`T` varies from 369 to 635 across the tree because the window is fixed but the sampling is
not, so the cubes differ in time *resolution*, not extent. Everything is resampled onto one
20 ms grid — the pipeline's own STFT hop — before anything is compared.

## What it does

| step | script | output |
|---|---|---|
| per-electrode timing features | `build_timing_table.py` | `outputs/timing/timing_table.csv` |
| group comparisons + figures | `make_rt_figures.py` | `outputs/figures/`, `outputs/tables/` |
| narrative walkthrough | `510_response_timing.ipynb` | — |

Timing features are measured on the mean 70–150 Hz trace over the response window
(0 → 5 s after GO): onset latency (first crossing of +1 dB sustained ≥100 ms), peak
latency, peak amplitude, centre of mass, suprathreshold duration, positive area, and the
mirrored suppression versions. Values are dB relative to the **pre-stimulus** baseline
(`cfg.baseline_w = (-0.6, -0.1)` before *stimulus* onset), so 0 dB is baseline and the
thresholds are absolute.

Each electrode is then joined to every group label it already carries — k-means and Ward
clusters on both feature sets, the convex-NMF leading component and its full weight
vector, and the pooling role — and the groups are compared.

## The statistic that matters

Electrodes within a patient share a brain, a reference, a montage and a response speed.
Kruskal–Wallis over pooled electrodes treats them as independent and returns p < 1e-10 for
every group and every condition, including ones that are plainly null. The headline test
here **shuffles group labels within each patient**, so it asks whether group membership
still orders electrodes in time *given that patient's own contacts*. On the first run:

| group | audio | picture | reading |
|---|---|---|---|
| Ward, concat_hg | 0.002 | 0.002 | 0.006 |
| k-means, concat_hg | 0.002 | 0.020 | 0.373 |
| convex-NMF lead | 0.002 | 0.008 | 0.002 |
| pooling role | 0.018 | 0.814 | 0.291 |

Kruskal–Wallis p was < 1e-10 for all twelve.

## Caveats that change the interpretation

- **The cubes are trial-averaged.** No per-trial data is persisted anywhere in the 05 tree,
  so nothing here can correlate a single trial's response time with a single trial's neural
  timing. Every comparison is *across electrodes*. Trial-level work needs 150 to persist
  per-trial HG first.
- **t = 0 is the GO cue, not speech onset.** There is no speech-onset event anywhere in
  this dataset. "Response portion" means "after the stimulus ended".
- **Onset latency is defined on a minority of electrodes** — roughly 580–790 of ~1200 per
  condition cross +1 dB for 100 ms. The rest have no onset, by construction, and drop out
  of that row rather than being scored as late.
- **Coverage is partial for four patients** in the RealTime tree: EL043 has no picture,
  EL044 only audio, PAT_3301 only picture, PAT_3965 only reading. This is why FIG RT.1
  shows PAT_3965 as incomplete — that figure describes the 05 tree, not the cohort.
  PAT_3965 has all three conditions in the 04 tree and is in every clustering run.
- **Only 3626 of 9241 rows carry a cluster label**, because the clustering cohort is the
  1266 electrodes that survived the activity gate with all three conditions present.
- **The 5 s cap** (`RT_MAX_POST_S`) means the average is over trials whose response fit
  inside it; slow blocks lose trials unevenly (EL046's reading median is 7.9 s).

## Reproduce

```bash
python build_timing_table.py
python make_rt_figures.py
```
