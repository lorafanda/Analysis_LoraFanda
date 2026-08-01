# Stage 04 — Power Pooling & Functional-Role Mapping: Methodology

## Motivation

Stages 02–03 of this pipeline asked *unsupervised* questions of the intracranial
data — how do trial-averaged time–frequency responses cluster, and what can a
decoder read out of them. Stage 04 turns the question around and runs a
**confirmatory, hypothesis-driven test**: given an *a-priori* model of what each
language process should look like in the warped trial timeline (auditory input,
visual input, lexical-semantic access, phonological encoding, speech-motor
output, …), **which electrodes express each signature, and where do those
electrodes sit on the brain?** It is the time–frequency region-of-interest (ROI)
counterpart to a data-driven cluster-based permutation test (Maris & Oostenveld,
2007): instead of letting the statistics discover the windows, we draw the
windows from neurolinguistic theory and ask each contact to pass or fail them.
The deliverable is a per-electrode functional label, rendered on the fsaverage
brain and cross-referenced against Yeo resting-state networks and Neurosynth
meta-analytic maps, plus an interactive web tool (poolv2) where the role
definitions can be redrawn and re-evaluated live.

---

## 1. Input data — the canonical ERSP samples

The pooling stage consumes the **same trial-averaged ERSPs** the clustering
stage built (`lf_dataset.prepare_dataset`), so every downstream comparison is on
identical samples. One sample = one **electrode × condition**, a 2-D
event-related spectral perturbation (ERSP) of shape **(129 frequencies × 300
time bins)**.

- **Conditions** (the three language tasks): `audio`, `picture`, `reading`.
- **Frequency axis**: 129 bins, linearly spaced 0 → `FMAX_HZ = 500 Hz`.
- **Time axis**: 300 bins, **time-warped 50 % stimulus / 50 % response**, so bin
  150 (the midpoint, 50 %) is the GO-cue / response onset. Left of 150 =
  perception, right of 150 = production.
- **Amplitude**: already **baseline-relative dB** (normalised to a −0.6…−0.1 s
  pre-stimulus baseline), so "compare against baseline" is baked in — a value of
  +3 dB means 3 dB above that electrode's own pre-stimulus power.

Loading yields `df_meta` (one row per electrode×condition, with the contact key
and source file path) and `X_3d` (N × 129 × 300). Non-neural channels are
dropped at this step (e.g. 8 841 of 9 018 samples retained in the current run).

---

## 2. Two data dimensionalities (`full` vs `ds`)

The same ERSP is matched at **two resolutions**, selected by the `USE_DS` knob in
notebook 460:

| Grid | Shape / condition | Freq axis | Time axis | Purpose |
|------|-------------------|-----------|-----------|---------|
| **`full`** | 129 × 300 | 129 linear bins, 0–500 Hz | 300 warped bins | Faithful baseline; blob discretisation; **HGA-timeseries visualisation** (smooth curves) |
| **`ds`** | 15 × 30 | 15 canonical bands, 1–400 Hz | 30 bins (anti-aliased) | Fast matching; **the grid the web tool serves** |

**Why downsample, and how.** The 15-band grid is the clustering pipeline's
"rawds" representation, produced by `lf_features.downsample_ersp_to_bands`:

1. **Frequency** → 15 contiguous bands (`FREQ_BANDS_15_TO_400HZ`): delta (1–4),
   theta (4–8), alpha (8–13), low/high beta (13–20, 20–30), low/high gamma
   (30–50, 50–70), HG low/mid/high (70–100, 100–130, 130–170), HFO
   (170–220, 220–270, 270–320, 320–360, 360–400). Each output row is the mean of
   the full-grid rows whose centre frequency falls in the half-open `[lo, hi)`
   band. Signal above 400 Hz is intentionally dropped.
2. **Time** → 30 bins per condition via **scikit-image anti-aliased `resize`**
   (Gaussian pre-filter before resampling — not a naïve block average), which
   suppresses aliasing when collapsing 300 → 30 bins.

> **Convergence note (fixed June 2026).** The interactive web cube must encode
> the *identical* ds grid the notebook matches on, or the same role definition
> gives different counts in the two places. Two subtleties were aligned: (i) the
> web cube now uses the same anti-aliased `resize` as the notebook (it previously
> block-averaged time, which kept peaks sharper and **over-counted**); (ii) both
> sides now operate on the same int8-quantised values (`CUBE_SCALE = 16`,
> 0.0625 dB steps), so the published role table and the live designer are
> **bit-identical** (0 role-match disagreements across all roles).

---

## 3. Concatenation — one map per contact

For each electrode that has **all three conditions**, the three ERSPs are
**stitched in time**: `[audio | picture | reading]`. This produces one map per
contact:

- `full`: **129 × 900** (300 + 300 + 300)
- `ds`: **15 × 90** (30 + 30 + 30)

Only complete contacts enter (current run: 2 856 contacts from 8 841 samples).
Concatenation is what lets a single role template reason *across* conditions —
e.g. "active to the spoken prompt **and** silent to the picture".

---

## 4. Functional roles — the templates

Roles are defined declaratively in `functions/roi_config_concatenated.py`. Each
role is a **conjunction of boxes** over the concatenated `[a|p|r]` grid. A box
is a rectangle in (condition, time-%, frequency-Hz) with a required **sign**:

| Sign | Meaning | Gate |
|------|---------|------|
| `pos` | activation | fraction of box cells **> thr** ≥ `frac` |
| `neg` | suppression | fraction **< −thr** ≥ `frac` |
| `zero` | **silent** | fraction with **|dB| > thr** must be **< frac** |
| `nonpos` | not-activating | fraction **> thr** must be **< frac** (suppression OR silence OK) |

A box is specified by `block` (audio/picture/reading), `t_pct` ([0–100] % of that
condition's warped axis), `f_hz` ([lo, hi] in Hz), and `sign`. Each role carries
its own **`thr`** (dB threshold) and **`frac`** (minimum fraction of cells that
must cross it). A contact **matches** a role only if **every** box passes (strict
logical AND); roles marked `match: "any"` pass on ≥ 1 box (used for the spectral
tags).

### 4.1 Three layers

The role list is factored into three orthogonal namespaces (the `layer` key):

- **Layer 1 — functional roles** (one per process hypothesis, each with its full
  multi-band fingerprint): `auditory`, `auditory_v2`, `visual_picture`,
  `visual_reading`, `picture_stim`/`pic_stim`, `lexical_semantic`,
  `heteromodal_convergence`, `maintenance`, `premotor planning`,
  `phonological_encoding`, `motor`, `motor_v2`, `auditory_feedback`,
  `deactivation`, `NN`.
- **Layer 2 — spectral tags** (single-signature, band-specific markers, `any`
  match): `tag_hga_activation`, `tag_beta_erd`, `tag_low_f_erd`,
  `tag_theta_tracking`, `ultra_hfa`. A contact that fires an ERD tag but matches
  **no** layer-1 HGA role is the "missed-by-a-high-gamma-only-filter" category.
- **Layer 3 — umbrellas** (coarse disjunctions): `stimulus_active`,
  `response_active`.

### 4.2 Multi-label + primary winner

Matching is **multi-label**: `roles_matched` lists every role a contact
satisfies. A single **primary** `role` is also chosen for the map colour, by:

1. **Layer priority** — a functional role (layer 1) always beats a spectral tag
   (layer 2), which beats an umbrella (layer 3). *(Fixed June 2026; previously
   "most boxes wins" let the 6-box `tag_hga_activation` steal primary from a
   2-box `auditory`.)*
2. **Most boxes** within the winning layer (the most specific / most constrained
   template).

Counts are reported two ways: **`n_matched`** (every role a contact shows — a
contact contributes to all of them) and **`n_primary`** (the single most-specific
winner).

### 4.3 Example role — `auditory`

```
boxes:
  audio   t 7–14 %   79–151 Hz   pos    (early HGA to the spoken prompt)
  picture t 3–24 %   52–185 Hz   zero   (silent to the visual picture)
thr = 1.5 dB, frac = 0.10
```

i.e. a contact is "auditory input cortex" if it shows high-gamma to the spoken
word **and** is silent in the same band while a picture is on screen.

---

## 5. Role design — the interactive web tool (poolv2)

`poolv2.html` is a browser tool (served from the GitHub repo, no backend) that
lets you **draw role boxes on a blank `[audio | picture | reading]` grid**, set
`thr` and `frac` with sliders, hit **Apply**, and watch every electrode recolour
live. It is the authoring surface for `roi_config_concatenated.py`.

- **Data**: a compact **cube** (`pool_cube/cube_i8.bin` + `cube_manifest.json`)
  holding every pooled contact's ds-grid ERSP as int8 (`round(dB × 16)`, ≈ 3.9 MB
  for 2 856 contacts). The manifest carries `n_freq=15`, `n_time=90`,
  `bins_per_block=30`, `scale=16`, `fmax_hz=500`, and the 15 `band_hz` edges.
- **Matching** (`boxPass`): for each drawn box, convert `t_pct` → time bins
  (`block·30 + round(pct/100·30)`) and `f_hz` → the band rows that overlap
  `[lo, hi]`, then compute the fraction-above-threshold gate — **the exact same
  arithmetic as the Python `box_expresses_raw`**, on the **exact same quantised
  values** as the notebook. What you see in the designer is what the published
  analysis records.
- **Export**: the "export role" button emits the box list as a
  `roi_config_concatenated.py` entry (`t_pct`/`f_hz` format) to paste back into
  the config.

A "Matching" toggle switches between the boxcar export (`pool_web/`, notebook
460) and a Gaussian-window robustness variant (`pool_web_gaussian/`, notebook
465).

---

## 6. Two matching engines (raw-dB vs discretised)

The same role templates can be evaluated two ways:

- **Raw-dB fraction gate** (`build_role_table_raw`, the poolv2 / web path, and
  the current default): the gate operates directly on the dB values — "≥ `frac`
  of the box's cells exceed `thr` dB". This is what the web designer does and
  what the published `contacts_pool.csv` uses.
- **Discretised −1/0/+1 gate** (`build_role_table` / `discretize_ersp`): first
  segment the ERSP into activation/suppression blobs with the **clustering-exact
  valley segmentation** (`CLUSTERING_SEG_KWARGS`, identical to stage 02), paint a
  ternary map, then a box "expresses" if ≥ half its cells carry the expected
  sign. This is the "faithful baseline" used for the full-resolution overlays in
  notebook 470.

---

## 7. HGA time-series & activation onset

To characterise *timing*, each role's (and each Neurosynth region's) member
contacts are averaged into a **high-gamma-activity (HGA, 70–150 Hz) time series**
per condition, computed on the **full-resolution** grid (300 bins/condition) for
smooth curves even though matching used the ds grid. From each curve an **onset**
is read as the first time the mean crosses a dB threshold
(`onset_thr_db ≈ 0.4–0.5`). Two correlation matrices (Pearson *r* between every
pair of role / region HGA curves, per condition) summarise which roles share a
temporal profile vs. fire at distinct times.

---

## 8. Electrode strength gradient

On the brain map, each electrode's colour is its role colour **graded by
response strength**. `hga_strength` = the contact's **peak HGA dB** (70–150 Hz,
across the full concatenated timeseries), min–max normalised **within each
layer-1 role** to [0, 1]. The exported colour is then linearly interpolated from
**white (weakest in that role) → full role colour (strongest)**, discretised to
10 steps (`_lerp_to_white`). So within, say, the auditory population you can see
at a glance which contacts carry the strongest high-gamma.

---

## 9. Anatomy — getting electrodes onto fsaverage in 3-D

Every electrode is placed in **fsaverage (MNI305) space** so all patients share
one brain. The coordinates are precomputed by the recon pipeline
(`02_FBM_Clustering/.../250_recon/`) and read from
`…/fsaverage/coords/<PATIENT>_contacts_fsaverage.csv`.

**Coordinate chain** (`lf_recon_shared.py`): native scanner/tkrRAS contact
coordinates → per-patient FreeSurfer `talairach.xfm` → **MNI305** → inverse
fsaverage talairach → **fsaverage tkrRAS**. The result space is recorded as
`fsaverage_tkrRAS_via_MNI305_affine`. Each row carries `x, y, z` (mm, fsaverage),
`hemi`, `is_wm` (white-matter depth flag), `is_cortical`, `dist_to_pial_mm`, and
the two Yeo labels.

**Rendering**: the web page (NiiVue) draws the fsaverage pial mesh and places one
sphere per electrode at its `x, y, z`, coloured by role (graded by strength).
Cortical vs depth contacts get different sphere radii (`is_cortical`). Electrodes
of the same colour are batched into one mesh for performance.

---

## 10. Yeo resting-state networks

Each contact is tagged with its **Yeo-2011 7-network and 17-network** membership
(`yeo7_network`, `yeo17_network`), used as an independent anatomical reference
("do my auditory-role contacts land in the expected networks?").

- **Source**: FreeSurfer `Yeo2011_7Networks_N1000` / `17Networks_N1000` `.annot`
  parcellations on fsaverage.
- **Assignment** (`lf_anatomy.build_yeo_cache`): build a `scipy.spatial.cKDTree`
  of the fsaverage pial vertices per hemisphere; for each electrode, query the
  **nearest pial vertex** (Euclidean) and read the network label there.
  White-matter depth contacts (`is_wm = 1`) are labelled `WhiteMatter` and
  skipped. The 7 networks are Visual, Somatomotor, Dorsal Attention, Ventral
  Attention, Limbic, Fronto-Parietal (control), Default; the medial wall and
  white matter are excluded from network summaries.

---

## 11. Neurosynth meta-analytic regions

A second, literature-based anatomical reference: where does each electrode fall
relative to **Neurosynth term maps**?

- **Maps**: `04_FBM_Pooling/neurosynth/<term>/…association-test_z_FDR…nii.gz`,
  one per term — **auditory, broca, lexical, motor control, phonological, recall,
  semantic, visual** (8 terms). Each voxel's z-score is the meta-analytic
  association between activation there and studies tagged with that term (FDR
  corrected), in **MNI152** space.
- **Assignment** (`assign_neurosynth_regions`): convert each electrode's
  fsaverage **MNI305 → MNI152** with the fixed linear affine `_MNI305_TO_152`,
  sample every term's z-map at that voxel, and **multi-label** the electrode with
  every term whose `z > z_thr` (default 0). `neurosynth_primary` = the
  highest-z term. Each region's member ERSPs are averaged into a "region card".

---

## 12. The a-priori window track (notebooks 410 → 420 → 430 → 490)

Parallel to role matching, an earlier track tests **time-zone responsiveness**
without per-process templates:

- **410 — zone discovery**: overlay every contact's blobs and the time-marginal
  count of active contacts; **hand-define** a few warped-time windows
  (perception, pre-articulation, articulation, reading), each as **both** a
  boxcar `[t_lo%, t_hi%]` and a Gaussian `(centre%, σ%)`. → `window_config.json`.
- **420 — pool & qualify**: pool ERSP power inside each window (time-weighted
  average; boxcar = uniform, Gaussian = centre-weighted, ±2σ support), for two
  feature sets — `hg` (70–150 Hz line) and `bands15` (each band separately). Each
  contact×condition×zone is **qualified** with the clustering high-activity gate
  *restricted to the window*: positive if `prop(>2.2σ) ≥ 0.02`, negative if
  `prop(<−3.0σ) ≥ 0.04` (both signs kept). Optional circular-shift permutation
  null.
- **430 — anatomy**: map qualifiers to Yeo-7/17 and Desikan-Killiany gyri
  (`aparc`, via MNE on fsaverage, cached); report per-zone **purity** (fraction
  in top region), **entropy**, and **spatial compactness** (mm spread,
  hemisphere-mirrored); render qualifiers on fsaverage (red = +, blue = −).
- **490 — results**: synthesises both tracks, with the key robustness check —
  **boxcar ≈ Gaussian** agreement means a zone effect is real and not an artefact
  of exactly where the window edge was drawn; divergence flags an edge-sensitive
  effect.

**Why two window shapes / feature sets** (490's rationale): the boxcar is the
straight hypothesis test but its hard edges cause spectral leakage; the Gaussian
taper tolerates trial-to-trial latency jitter (a temporal average-pooling kernel;
the principled generalisation is multitaper estimation). High-gamma is the
canonical task-response marker; the 15-band set catches band-specific effects.

---

## 13. Published outputs

| File | Content |
|------|---------|
| `pool_web/contacts_pool.csv` | per-electrode: patient, contact, name, hemi, x, y, z, is_cortical, role, roles_matched, Yeo labels, **strength-graded colour** |
| `pool_web/pool_index.json` | per-role `n` (multi-label) and `n_primary` (winner) |
| `pool_web/role_info.json` | per-role description, colour, box template, onset, card image |
| `pool_web/role_cards/<role>.png` | a representative member's `[a|p|r]` ERSP with the role's boxes drawn |
| `pool_cube/cube_i8.bin` + `cube_manifest.json` | the int8 ds-grid ERSP the web designer matches against |
| `neurosynth/…` | per-contact term labels + region cards |
| `pool_web_gaussian/…` | the Gaussian-window robustness variant (notebook 465) |

The web page (`poolv2.html`) fetches `pool_web/` + `pool_cube/` straight from the
committed repo via raw-GitHub, so **committing + pushing these outputs publishes
the analysis**.

---

### Key references
Pfurtscheller & Lopes da Silva (1999, ERD/ERS) · Maris & Oostenveld (2007,
cluster-based permutation) · Harris (1978, windows/tapers) · Thomson (1982,
multitaper) · Yeo et al. (2011, 7/17 networks) · Yarkoni et al. (2011,
Neurosynth) · Hamilton et al. (2018) & Forseth et al. (2018, electrode response
typing + a-priori windows).
