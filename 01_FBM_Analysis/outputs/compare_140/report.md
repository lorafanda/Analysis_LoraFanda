# What the 2026-09-18 rerun changed — 140's two output trees, contact by contact

Written 2026-09-24. Everything here is measured, not remembered; the scripts are
`compare_140_trees.py`, `attribute_140_changes.py`, `compare_140_hg.py` and
`make_compare_140_figures.py` in `01_FBM_Analysis/`, and every number is in the TSVs
beside this file.

---

## The headline

**The cubes below 400 Hz are not the old cubes cropped. They were computed from a
differently filtered signal.** The adaptive notch is given `cfg.fmax`, so it only
considers mains harmonics up to it: at 500 Hz it removed harmonics at **400–483 Hz** that
at 400 Hz it no longer removes. The notch is applied to the **time series** (method
`interp`, which rebuilds the signal through the FFT), so dropping a harmonic above 400 Hz
changes the signal at every frequency — including the bins that were kept.

Of 9,529 contact-conditions present in both trees, **430 (4.5 %) agree to within 0.01 dB**
once the old cube is cropped to the new axis. None is bit-identical. The rest differ by a
median RMSE of 0.03–1.4 dB depending on the patient.

This is not an error — the new tree is the correct one, the old one is superseded. It is
a statement about what "we only lowered the ceiling" actually did.

---

## How it was checked — three passes, because one number is not a check

| pass | what it reads | n | independent of |
|---|---|---|---|
| 1 · cubes | `ERSP_matrix/<cond>/*.npy`, the trial-averaged cube | 9,529 pairs | — |
| 2 · halves | `ERSP_halves/<cond>/*_half1|2.npy`, the odd/even split cubes | 19,058 pairs | different files, written by a different line of the notebook |
| 3 · HG | `04_ersp_LM/<pid>/LM/HG/<cond>/*.png`, the trial rasters | 11,510 pairs by size, 442 by pixel | a different medium entirely; blind to the frequency ceiling |

**Passes 1 and 2 agree at Spearman ρ = 0.999** (FIG D.1 D). Any patient flagged by only
one of them would be a bug in that pass; none is.

The comparison always crops the old cube (129 bins, 0–500 Hz) to the new one's height
(103 bins, 0–398.4 Hz). Both trees run the same `nperseg` / `nfft` / fs, so bin *k* is the
same 3.90625 Hz band in both — the crop is exact, and every difference that remains is
something else.

---

## Which patients changed, and by how much

Median RMSE over a patient's contacts, and the date its OLD cubes were written. Full
table: `attribution.tsv`; per patient × condition: `summary_cubes_patient_condition.tsv`.

| patient | old run | median RMSE (dB) | worst r | notch harmonics lost above the new ceiling | what it is |
|---|---|---|---|---|---|
| EL045 | 2026-09-07 | **1.384** | 0.16 | 27 (450 Hz) | notch reconstruction, broadband ≈ 1 dB, rising to 2.6 dB at 398 Hz |
| EL043 | 2026-08-22 | **1.120** | −0.04 | audit missing | **new picture triggers** (18 June recording) + everything since August |
| EL046 | 2026-08-25 | 0.713 | 0.69 | audit missing | broadband, 1/f-shaped |
| PAT_6704 | 2026-09-15 | 0.570 | 0.69 | 114 (to 500 Hz), 40 decisions flipped | mains comb + the MicroEPI reference fix |
| EL040 | 2026-08-21 | 0.525 | 0.76 | audit missing | textbook mains comb |
| EL037 | 2026-08-21 | 0.493 | 0.54 | audit missing | mains comb |
| EL036 | 2026-08-21 | 0.434 | 0.86 | audit missing | mains comb |
| EL042 | 2026-08-21 | 0.402 | 0.87 | audit missing | broadband |
| EL048 | 2026-09-07 | 0.367 | 0.90 | 144 (to 483 Hz, 16.67 Hz railway comb) | smooth rise with frequency, 1e-6 → 1 dB |
| EL038 | 2026-08-21 | 0.308 | 0.93 | audit missing | flat 0.4 dB offset in **audio only** → trials |
| PAT_5533, PAT_6619 | 2026-08-21 | 0.31, 0.30 | 0.92 | audit missing | mains comb |
| EL033 | 2026-08-26 | 0.244 | 0.97 | audit missing | comb at 100 / 300 Hz |
| EL051 | 2026-09-17 | 0.200 | 0.96 | 3 (450 Hz) | smooth, low |
| EL044, EL034, PAT_6953 | Aug | 0.12–0.16 | 0.98 | — | small |
| PAT_6854 | 2026-09-15 | 0.076 | 0.87 | 120, 7 flipped | comb + reference fix |
| PAT_3415 … PAT_2868 | 2026-08-21 | 0.02–0.08 | ≥ 0.99 | — | small |
| **PAT_5515** | 2026-09-15 | **0.000** | 0.89 | 84, 2 flipped | 148 of 444 within 0.01 dB — exactly one condition untouched |
| **EL052** | 2026-09-17 | **0.000** | 1.00 | 3 | 282 of 312 within 0.01 dB; max 0.055 dB — the control |

EL052 is the control that proves the method: its old cubes are from the day before the
rerun, nothing else about it changed, and its difference spectrum is a smooth
1e-8 → 1e-4 dB ramp — the pure numerical footprint of removing one 450 Hz notch.

---

## Where the change sits — FIG D.2 (frequency) and FIG D.3 (time)

`figures/D2_why_frequency.png` plots mean |new − old| per frequency bin for every patient,
log scale, with 50 Hz and its harmonics marked. Three signatures, and they are easy to
tell apart:

- **A comb on the grey lines** — EL040, EL037, EL036, PAT_5533, PAT_6619, PAT_6704,
  PAT_6854, PAT_3415, EL043: the notch. The difference is 1–2 orders of magnitude larger
  at the harmonics than between them.
- **A smooth monotonic rise with frequency** — EL048, EL052, PAT_5515: the
  spectrum-interpolation notch's global reconstruction, largest next to the harmonics that
  are no longer treated.
- **A flat lift across every frequency, in one condition only** — EL038 audio, PAT_3066
  reading, PAT_3780 / PAT_3965 picture: the set of trials that went into the average.

`figures/D3_why_time.png` does the same over the warped axis. A difference confined to one
half is the response window or the trial set; one spread evenly is preprocessing. Across
the cohort the stimulus and post halves change equally (0.177 vs 0.181 dB), which is what
a preprocessing change looks like.

---

## What appeared and disappeared

944 contact-conditions exist in only one tree — 942 only in the old, 2 only in the new
(`cubes_missing.tsv`, FIG D.1 C):

- **PAT_6684 83** — G-05, removed from the dataset on 09-15.
- **PAT_6704 55, PAT_6854 38, PAT_6953 33** — the MicroEPI bad lists and the reference fix.
- **EL051 26, EL052 21** — the bad-electrode lists added on 09-17/18.
- **EL030 13, EL035 12, EL043 7, EL034 / EL038 6, EL042 5** — bad lists and, for EL043,
  the rebuilt picture block.
- **EL045 +1 / −1** — the only patient that gained a contact.

---

## The HG rasters — pass 3

11,510 HG figures exist in both trees; **33 are byte-identical**. That number is
misleading on its own, and the pixel pass says why: the HG raster was **rewritten on
2026-09-09** to draw every excluded trial with its reason. So a patient whose old tree
predates that rewrite has a cosmetically different figure whatever the data did.

The split is clean:

- old tree **before** 09-09 → median pixel difference **0.06–0.09** (the rewrite), every
  patient from EL030 to PAT_3455.
- old tree **after** 09-09 → **PAT_6704 0.022, EL051 0.005, PAT_6854 0.002, PAT_5515 0.000,
  EL052 0.000** — i.e. the trials did not move for these patients, only the spectra did.

That is the separator the pass exists for: for every patient rerun since 09-09, the change
in the cubes is preprocessing, not data.

---

## The case figures

`figures/cases/` holds 60 figures, the two most-changed contacts of each patient. Each
one shows, in a row: the OLD cube at its full 0–500 Hz, the OLD cube cropped to the new
axis, the NEW cube, and their difference — and underneath, the OLD and NEW HG rasters.

The one to look at first is **`EL043_picture_sSMG8.png`** (r = 0.03, RMSE 1.69 dB): the two
HG rasters are not the same data at all. Old: 35 trials, stimulus durations to 8 s. New:
53 trials of which 25 kept, durations ≤ 5 s, 28 excluded and labelled. That is the 18 June
picture block replacing the bad 17 June one — the largest single change in the cohort, and
a correct one.

---

## What this means for the analysis

1. **The cohort's cubes are internally consistent** — every patient in the new tree comes
   from the 09-18 code, and the 09-21/22 reruns use the same code. Nothing mixes trees.
2. **Any v8-era number is not comparable at the dB level** to a v9-era one, even below
   400 Hz. Between the notch change and the bad lists, the median contact moved by
   0.03–1.4 dB.
3. **If the 400 Hz ceiling was meant to change only the stored axis, it did not.** Keeping
   the notch harmonic list at 500 Hz while storing 400 Hz would have separated the two
   decisions; that is a one-line change (`fmax` passed to the notch functions in 140's
   cells, independent of `cfg.fmax`) and it is a decision, not a bug fix.
