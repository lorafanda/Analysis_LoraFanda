# Reading the classification outputs — definitions, equations, what to look for

A reference for every plot/metric the `03_FBM_Classifying` pipeline writes per run
(`outputs/classification/<task>/<variant>/<classifier>/runs/<id>/`). The live narrative
that *displays* these is `390_results.ipynb`.

---

## 0. Mental model (read first)

Each run trains **logistic regression** or **random forest** to predict a label (condition,
or Yeo network) from an electrode's ERSP features, evaluated by **nested GroupKFold by
patient** — every test prediction comes from a model that never saw that patient. So all
numbers estimate **generalization to a new patient**, and the effective sample size is
**#patients (~20), not #electrodes**.

**Chance levels** (the line everything is judged against):

| task | classes | chance (balanced acc) |
|---|---|---|
| Condition | audio / picture / reading | **0.333** |
| Parcellation Yeo-7 | 7 networks | **1/7 ≈ 0.143** |
| Parcellation Yeo-17 | 17 networks | **1/17 ≈ 0.059** |

> **Golden rule:** *above-chance ≠ strong ≠ meaningful.* Always read the headline number, the
> confusion structure, the per-class stars, **and** the amplitude triad together.

Notation below: **K** classes; for class *k*, TP/FP/FN are its true-positive / false-positive
/ false-negative counts; **N** = total samples; **B** = number of permutations.

---

## 1. Headline scalars (`metrics.json`)

| metric | equation | read it as |
|---|---|---|
| **Recall_k** (sensitivity) | `TP_k / (TP_k + FN_k)` | of true-*k* electrodes, fraction caught |
| **Precision_k** | `TP_k / (TP_k + FP_k)` | when we predict *k*, how often correct |
| **F1_k** | `2·P_k·R_k / (P_k + R_k)` | harmonic mean of the two |
| **Balanced accuracy** ← headline | `BA = (1/K) Σ_k Recall_k` | mean per-class recall; robust to imbalance. Compare to `1/K` |
| **Macro-F1** | `(1/K) Σ_k F1_k` | unweighted per-class F1 |
| **Accuracy** (not headline) | `Σ_k TP_k / N` | misleading under imbalance — a model that always predicts the majority class scores high |

**Balanced-accuracy 95% CI** = bootstrap over patients: resample patients with replacement,
recompute BA, take the **[2.5, 97.5] percentiles**.
*Look for:* **lower CI bound above chance** ⇒ reliably above chance. CI straddling chance ⇒ not.

---

## 2. `confusion_matrix.png` / `.csv` — who gets mixed up

Row-normalized: `C[i,j] = (#true-i predicted-j) / (#true-i)`. Rows = truth, cols =
prediction; **the diagonal = recall per class**; each row sums to 1.

*Look for:*
- **Bright diagonal** → clean separation.
- **Bright off-diagonal block** → systematic confusion between two classes (e.g. audio↔reading look alike).
- **One bright column** → the model is dumping everything into one class (collapse) — check this whenever BA is near chance but raw accuracy looks "ok."

---

## 3. `per_class_strength.png` + `per_class_metrics.csv` — *the most important panel*

Bars = **Recall_k** with bootstrap **CI whiskers**, a dashed **chance line (1/K)**, and
**significance stars**.

- **ROC-AUC (one-vs-rest)_k** = probability the model ranks a random true-*k* electrode above
  a random non-*k* one (0.5 = chance, 1 = perfect). Threshold-free; complements recall.
- **Per-class permutation p** on recall: shuffle labels B times,
  `p_k = (1 + #{Recall_k(shuffled) ≥ Recall_k(obs)}) / (B + 1)`.
- **FDR-corrected** (Benjamini–Hochberg over the K classes): sort `p_(1) ≤ … ≤ p_(m)`,
  adjusted `p_(i) = min_{j ≥ i} (m/j)·p_(j)`; stars use `perm_p_fdr`.
  `*** p<.001 · ** p<.01 · * p<.05 · ns`.

*Look for:* **which classes actually separate.** A bar above the chance line *with a star and a
CI clearing chance* is a real, recoverable class. `ns` classes collapse into neighbours — the
summary BA is being carried by a subset of classes (common for imbalanced Yeo).

---

## 4. `permutation_null.png` (`permutation_null.json`) — is the *overall* result luck?

Histogram = balanced accuracy under **B label shuffles** (the null); red line = observed BA.
`p = (1 + #{BA(null) ≥ BA(obs)}) / (B + 1)`.

*Look for:* observed line **far in the right tail**, `p < .05`. If it sits inside the null
bulk, the decode is chance regardless of the raw BA value. (Skipped when `N_PERM=0`.)

---

## 5. `feature_importance.png` (+ `_by_band/_by_condition/_by_time.csv`) — *what drove it*

- **LR:** signed standardized coefficients per class; aggregated importance =
  `mean_k |coef_k|` per feature, summed by band / condition / time.
- **RF:** impurity importance + **permutation importance** (drop in BA when a feature is
  shuffled). Auto-skipped above 1200 features (`full_300`) — trust the LR coefs there.

*Look for:* which **band** (HG vs low-freq), **time bin** (first half = perception, second =
response — see §7), and **condition** (Task B) carry the signal.
**Caveat:** with correlated features + strong regularization (e.g. `C=0.01`), importance is
diffuse and unstable — read it as "where roughly," not a precise ranking.

---

## 6. `class_feature_heatmap.png` / `.csv` — per-class fingerprint

Rows = classes; cols = **band** (and **condition×band** for parcellation); colour =
**per-class mean of the z-scored feature** = `mean over class-k electrodes of z(feature)`
(red = above-average response, blue = below).

*Look for:* read **down a column** (which classes light up in this band) and **across a row** (a
class's spectral signature). For parcellation: does one condition's block dominate a network,
or all three?

---

## 7. `class_ersp_profile.png` — the real spectro-temporal signature

Per class, the **mean full-spectrum ERSP** (15 bands × time); for parcellation the three
conditions are **concatenated** `[audio | picture | reading]`. A dashed **grey line at 50% of
each block = stimulus-offset / response-onset** (ERSPs are time-warped 50% sensing / 50%
response): **left half = perception**, **right half = production/response**.

*Look for:* a warm high-gamma patch in the **left** half = stimulus-driven; in the **right**
half = response/output. Shown even on HG runs, so you see the whole map the HG line is a slice
of. Tells you *why* a class is (or isn't) separable, and whether it's just **amplitude**
(uniformly hot) vs a real **pattern**.

---

## 8. `coef_heatmap.png` / `.csv` (LR only) — what the *model* uses

Rows = classes, cols = bands; colour = signed LR weight. Cell > 0 (red) = this band pushes an
electrode **toward** that class; < 0 (blue) = away.

*Look for:* contrast with §6 — the profile shows the *data's* signature, this shows what the
*linear model leveraged* to separate (after seeing all features jointly).

---

## 9. Cross-run reading (`390_results.ipynb`)

- **Amplitude triad** (`full_* → full_*_rn → m101_*`): if `continuous ≈ row-normed ≈
  discretized`, the **pattern** carries it; if `continuous ≫ the others`, you decoded
  **loudness/SNR**, not a code. *Read this before believing any "pattern" claim.*
- **Matched pairs** (`full_300 vs hg_300`, `full_30 vs hg_30`): full-spectrum beating HG at the
  **same time grid** = frequencies beyond HG add information.
- **Time grids** (`*_300 vs *_30`): the time-resolution effect — but freq/time are STFT-coupled,
  so treat as suggestive, not proof.

---

## 10. 30-second "is it real *and* meaningful?" checklist

1. **Permutation `p < .05`?** (not luck) →
2. **Lower CI bound above chance?** (reliable) →
3. **Confusion diagonal bright, not a one-column collapse?** (genuine separation) →
4. **Which classes carry the stars?** (per-class, not just the average) →
5. **Survives `m101` / row-norm?** (pattern, not amplitude) →
6. **ERSP profile + importance physiologically sensible?** (perception vs response timing,
   plausible bands).
