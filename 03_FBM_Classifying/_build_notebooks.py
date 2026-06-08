#!/usr/bin/env python3
"""Generate the 03_FBM_Classifying notebooks as valid nbformat-v4 JSON.

Run once with stdlib python3 (no scientific stack needed):
    python3 _build_notebooks.py
This file is a build helper, not part of the pipeline; safe to delete after.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def cell(kind, text):
    src = text.strip("\n") + "\n"
    lines = src.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": lines}


def write_nb(name, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = HERE / name
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", path)


# ============================================================
# 310 — data-prep
# ============================================================
nb310 = [
cell("markdown", r"""
# 310 — Classification data-prep & feature cache

Builds the classification-ready feature matrices **once** and caches them to disk so the
experiment notebooks (`320`, `330`) load instantly and never re-walk the ERSP tree.

**Source** — the same per-electrode ERSP `.npy` files the clustering pipeline uses:
`01_FBM_Analysis/outputs/04_ersp_LM_RAWONLY/<pid>/LM/ERSP_matrix/<cond>/`.

**Four feature variants — a nested, time-matched family** so "full spectrum vs
high-gamma" is a *fair* test (HG and full share a time grid; only frequency changes):
| variant | what | freq × time | dims |
|---|---|---|---|
| `full_300` | full spectrum, 15 bands | 15 × 300 | 4500 |
| `hg_300` | single 70–150 Hz line | 1 × 300 | 300 |
| `full_30` | full spectrum, downsampled time | 15 × 30 | 450 |
| `hg_30` | single HG line, downsampled time | 1 × 30 | 30 |

Valid contrasts are the **matched pairs**: `full_300 vs hg_300` and `full_30 vs hg_30`
(frequency content, time held fixed); `full_300 vs full_30` isolates time resolution.
⚠️ Frequency and time go hand in hand — the STFT couples their resolution, so even
matched grids aren't a perfectly clean separation (stated, not hidden).

**What it caches**
- **Condition task** (gated): one sample per high-activity electrode × condition.
- **Parcellation task**: one sample per electrode = `audio ⊕ picture ⊕ reading`
  concatenated, for each variant × {Yeo-7, Yeo-17}. An electrode is kept iff it has all
  three conditions, is high-activity in ≥1 condition, and sits in a real Yeo network
  (medial-wall / white-matter / unknown contacts are dropped).

Only patients with **all three** conditions enter either task. Run this first, then 320 & 330.
"""),
cell("code", r"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path('..').resolve()))
from functions import lf_classify as C

# Server UNC path first, local relative path as fallback (mirrors 02's notebooks).
INPUT_DIR = Path(r'\\nasac-m2.unige.ch\m-HumanNeuronLab\ANALYSIS\FLM\Analysis_LoraFanda\01_FBM_Analysis\outputs\04_ersp_LM_RAWONLY')
if not INPUT_DIR.exists():
    INPUT_DIR = Path('../01_FBM_Analysis/outputs/04_ersp_LM_RAWONLY').resolve()

print('INPUT_DIR :', INPUT_DIR, '| exists:', INPUT_DIR.exists())
print('CACHE     :', C.DATASET_CACHE)
print('COORDS    :', C.COORDS_DIR, '| exists:', C.COORDS_DIR.exists())
print('variants  :', C.VARIANTS)
"""),
cell("markdown", r"""
## 1 — Load the full ungated dataset
Walks every electrode × condition ERSP for patients that have all three conditions. The
high-activity flag is computed per sample (so both tasks derive from this one object).
The heavy walk is itself cached by `prepare_dataset`, so re-running is cheap.
"""),
cell("code", r"""
df_meta, X_3d = C.prepare_full_dataset(INPUT_DIR)
print('samples:', len(df_meta), '| X_3d:', X_3d.shape)
print('patients:', sorted(df_meta.patient_id.unique()))
df_meta.head()
"""),
cell("markdown", r"""
## 2 — Condition task — cache the 4 feature variants
Gated electrode × condition samples, labelled by condition.
"""),
cell("code", r"""
for v in C.VARIANTS:
    X, y, groups, meta, cols = C.build_condition_arrays(df_meta, X_3d, v)
    d = C.save_arrays('condition', None, v, X, y, groups, meta, cols)
    print('  saved ->', d)
"""),
cell("markdown", r"""
## 3 — Parcellation task — cache 4 variants × {Yeo-7, Yeo-17}
One sample per electrode; features = the three conditions concatenated in fixed order
`[audio, picture, reading]`. Label = the electrode's Yeo network.
"""),
cell("code", r"""
for n_net in (7, 17):
    for v in C.VARIANTS:
        X, y, groups, meta, cols = C.build_parcellation_arrays(
            df_meta, X_3d, v, n_networks=n_net)
        d = C.save_arrays('parcellation', f'yeo{n_net}', v, X, y, groups, meta, cols)
        print('  saved ->', d)
"""),
cell("markdown", r"""
## 4 — Summary
The cache under `outputs/_dataset/classification/` now holds every (task × variant) matrix.
320 and 330 read these directly — no need to re-run 310 unless the upstream ERSPs or the
high-activity / Yeo filters change.
"""),
cell("code", r"""
rows = []
for v in C.VARIANTS:
    X, y, g, m, c = C.load_arrays('condition', None, v)
    rows.append(('condition', '-', v, str(X.shape), len(set(y)), len(set(g))))
for n_net in (7, 17):
    for v in C.VARIANTS:
        X, y, g, m, c = C.load_arrays('parcellation', f'yeo{n_net}', v)
        rows.append(('parcellation', f'yeo{n_net}', v, str(X.shape), len(set(y)), len(set(g))))
pd.DataFrame(rows, columns=['task', 'target', 'variant', 'X.shape', 'n_classes', 'n_patients'])
"""),
]

# ============================================================
# 320 — condition classification
# ============================================================
nb320 = [
cell("markdown", r"""
# 320 — Condition classification (audio / picture / reading)

Decodes the **stimulus condition** from a single electrode's spectro-temporal response.
3 classes, one sample per high-activity electrode × condition, run for **each of the 4
feature variants × each classifier** (logistic regression + random forest) = 8 experiments.
Compare the **matched pairs** `full_300 vs hg_300` and `full_30 vs hg_30` (see 390).

Every metric below comes from **nested GroupKFold by patient** — the outer fold holds out
whole patients for testing, the inner fold tunes hyper-parameters, and no patient ever
appears in both train and test. That makes the numbers an estimate of *generalisation to a
new patient*, not memorisation of these ones.

**How to read each run** (full narrative + figures in `390_results.ipynb`):
- **Balanced accuracy vs chance (0.333)** — headline separability, robust to imbalance.
- **Permutation p** — is balanced accuracy above a patient-shuffled-label null?
- **Confusion matrix** — which conditions get mixed up with which.
- **Per-class strength** — recall ± bootstrap CI, with FDR significance stars.
- **Feature importance** — which bands / time bins drove the separation.
"""),
cell("code", r"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path('..').resolve()))
from functions import lf_classify as C

# ---------------- config knobs ----------------
CLASSIFIERS  = ('logreg', 'rf')
OUTER_SPLITS = 5      # outer GroupKFold (held-out test patients)
INNER_SPLITS = 3      # inner GroupKFold (hyper-parameter tuning)
N_PERM       = 200    # label-permutation null reps (set 0 to skip; rf is the slow part)
N_BOOT       = 1000   # bootstrap reps for per-class CIs (cheap)
RANDOM_STATE = 42
print('classifiers:', CLASSIFIERS, '| outer/inner:', OUTER_SPLITS, INNER_SPLITS,
      '| n_perm:', N_PERM)
"""),
cell("markdown", r"""
## Run all condition experiments
4 variants × 2 classifiers = 8 runs, each saved under
`outputs/classification/condition/<variant>/<classifier>/runs/<id>/`.
"""),
cell("code", r"""
# Full-spectrum (15 bands x 300 time) drives the per-class ERSP *response profile*
# shown for every run, regardless of which variant the classifier used — so HG runs
# still show the full spectrogram they are a slice of. n_cond=1 (class == condition).
Xf, yf, gf, mf, cf = C.load_arrays('condition', None, 'full_300')
profile = {'X': Xf, 'n_time': 300, 'n_cond': 1}

manifests = []
for v in C.VARIANTS:
    X, y, groups, meta, cols = C.load_arrays('condition', None, v)
    for clf in CLASSIFIERS:
        m = C.run_experiment('condition', v, clf, X, y, groups, cols, meta,
                             profile=profile,
                             outer_splits=OUTER_SPLITS, inner_splits=INNER_SPLITS,
                             n_perm=N_PERM, n_boot=N_BOOT, random_state=RANDOM_STATE)
        manifests.append(m)
print('\ndone:', len(manifests), 'runs')
"""),
cell("markdown", r"""
## Summary table
"""),
cell("code", r"""
df = C.list_runs()
df = df[df.task == 'condition']
df[['variant', 'classifier', 'balanced_accuracy', 'chance_level',
    'macro_f1', 'permutation_p']].round(4)
"""),
]

# ============================================================
# 330 — parcellation classification
# ============================================================
nb330 = [
cell("markdown", r"""
# 330 — Anatomical parcellation classification (Yeo-7 & Yeo-17)

Decodes **which Yeo functional network** an electrode sits in, from its response profile
**concatenated across all three conditions** (`audio ⊕ picture ⊕ reading`). Run for each of
the 4 feature variants × {Yeo-7, Yeo-17} × each classifier = 16 experiments.

The classes are imbalanced (some networks have far more electrodes than others), so
**balanced accuracy** and **macro-F1** are the headline metrics and `class_weight='balanced'`
is used throughout. Same nested-GroupKFold-by-patient protocol as 320 — so a network is only
"decodable" if it generalises across patients, not because a couple of patients dominate it.

See `390_results.ipynb` for the figures and how to read them.
"""),
cell("code", r"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path('..').resolve()))
from functions import lf_classify as C

# ---------------- config knobs ----------------
YEO_NETWORKS = (7, 17)
CLASSIFIERS  = ('logreg', 'rf')
OUTER_SPLITS = 5
INNER_SPLITS = 3
N_PERM       = 200
N_BOOT       = 1000
RANDOM_STATE = 42
print('yeo:', YEO_NETWORKS, '| classifiers:', CLASSIFIERS, '| n_perm:', N_PERM)
"""),
cell("markdown", r"""
## Run all parcellation experiments
2 Yeo granularities × 4 variants × 2 classifiers = 16 runs, saved under
`outputs/classification/parcellation_yeo{7,17}/<variant>/<classifier>/runs/<id>/`.

The per-class **response profile** is the full ERSP of each network, **concatenated
across audio+picture+reading** (a spectrogram triptych, not a single line), with a
grey stim→response boundary per condition.
"""),
cell("code", r"""
manifests = []
for n_net in YEO_NETWORKS:
    key = f'yeo{n_net}'
    # full-spectrum, 3 conditions concatenated -> the per-class ERSP triptych
    Xf, yf, gf, mf, cf = C.load_arrays('parcellation', key, 'full_300')
    profile = {'X': Xf, 'n_time': 300, 'n_cond': 3, 'cond_names': list(C.CONDITIONS)}
    for v in C.VARIANTS:
        X, y, groups, meta, cols = C.load_arrays('parcellation', key, v)
        for clf in CLASSIFIERS:
            m = C.run_experiment(f'parcellation_{key}', v, clf, X, y, groups, cols, meta,
                                 n_networks=n_net, profile=profile,
                                 outer_splits=OUTER_SPLITS, inner_splits=INNER_SPLITS,
                                 n_perm=N_PERM, n_boot=N_BOOT, random_state=RANDOM_STATE)
            manifests.append(m)
print('\ndone:', len(manifests), 'runs')
"""),
cell("markdown", r"""
## Summary table
"""),
cell("code", r"""
df = C.list_runs()
df = df[df.task.str.startswith('parcellation')]
df[['task', 'variant', 'classifier', 'balanced_accuracy', 'chance_level',
    'macro_f1', 'permutation_p']].round(4)
"""),
]

# ============================================================
# 390 — results narrative
# ============================================================
nb390 = [
cell("markdown", r"""
# 390 — Classification results (read me)

This is the **results page** for `03_FBM_Classifying`. It doesn't compute anything heavy —
it pulls the artifacts that 320 and 330 wrote and lays them out with the story and a guide
to reading each figure. Run 310 → 320 → 330 first, then run this top-to-bottom.

---

## What was asked

Two supervised questions, both built on the per-electrode ERSPs from `01_FBM_Analysis`:

1. **Condition decoding** — from one electrode's spectro-temporal response, can we tell the
   trial was **audio**, **picture**, or **reading**? *(3 classes; one sample per
   high-activity electrode × condition.)*
2. **Parcellation decoding** — from an electrode's profile **across all three conditions
   concatenated**, can we tell which **Yeo network** it sits in? *(7- or 17-class; one
   sample per electrode.)*

## The comparison is built to be *fair* — matched, nested feature variants

The point is to ask **"does the full spectrum beat high-gamma alone?"** For that to mean
anything, HG and full-spectrum must differ **only** in frequency content — same time grid,
same electrodes, same pipeline. So the four variants form two **matched pairs**:

| pair | full spectrum | high-gamma | what it isolates |
|---|---|---|---|
| 300-time | `full_300` (15 bands × 300) | `hg_300` (1 line × 300) | frequency content, time fixed |
| 30-time | `full_30` (15 bands × 30) | `hg_30` (1 line × 30) | frequency content, time fixed |

Read **`full_* vs hg_*` at the *same* time length** — never `hg_300` (fine time) against
`full_30` (coarse time), which is what an earlier version did and which tangles frequency
with time. `full_300 vs full_30` (or `hg_300 vs hg_30`) isolates the time-resolution effect.

> ⚠️ **Frequency and time go hand in hand.** The STFT has a fixed time–frequency resolution
> trade-off, so HG-with-fine-time vs full-with-coarse-time can never be a *perfectly* clean
> separation. Matching the time grid removes the gross confound; the residual coupling is a
> genuine caveat — treat differences between the 300 and 30 families as suggestive, not proof.

Two classifiers per variant: **logistic regression** (linear, interpretable) and **random
forest** (non-linear).

## How it was validated — nested GroupKFold by patient

Patients are the unit of splitting. The **outer** loop holds out whole patients as the test
set; the **inner** loop (on the remaining patients) tunes hyper-parameters. No patient is ever
in both train and test, so every number estimates **generalisation to a new patient**. Feature
scaling is fit on training folds only. Class imbalance is handled with `class_weight='balanced'`,
which is why **balanced accuracy** (not raw accuracy) is the headline.

## How to read each output

| Output | What it is | How to read it |
|---|---|---|
| **Balanced accuracy** | mean per-class recall, out-of-fold | compare to **chance = 1/#classes**; the 95% CI is bootstrapped over patients |
| **Permutation p** | balanced accuracy vs a patient-shuffled-label null | `p < 0.05` ⇒ separability is above chance, not luck |
| **Confusion matrix** | row-normalised: rows = true class, cols = predicted | bright diagonal = clean separation; bright off-diagonal = systematic confusion between two classes |
| **Per-class strength** | recall ± bootstrap CI per class | bars above the dashed chance line with `*` (FDR-corrected) are the classes that genuinely separate; `ns` = not distinguishable |
| **Feature importance** | what drove the model | which **condition** (Task B), **frequency band**, and **time bin** carried the signal |
| **Class × feature heatmap** | rows = classes (regions / conditions), cols = frequency bands × condition; colour = per-class mean response | read **down a column** to see which classes light up in a band; **across a row** to see a class's spectral fingerprint. Red = above-average, blue = below |
| **Per-class ERSP profile** | the **full-spectrum** ERSP of each class, **concatenated across audio+picture+reading** (a spectrogram triptych — *not* a single line) | this is the real response signature; for parcellation each Yeo network gets `[audio｜picture｜reading]` side by side. **Always full-spectrum**, even on HG runs, so you see the whole map HG is a slice of |

**Reading the ERSP profile — the stim→response structure.** Each ERSP is time-warped so the
**first half is the stimulus** (viewing / hearing) and the **second half is the response**.
The **dashed grey line at 50%** of every condition block marks that boundary: left of it =
sensing, right of it = response/production. So "an early-vs-late patch" literally means
"during perception vs during the response."
| **Coefficient heatmap** (LR) | rows = classes, cols = bands; colour = signed linear weight | what the model *uses* (vs the raw profile above): which band pushes an electrode **toward** (red) or **away from** (blue) each class |

`*** p<.001 · ** p<.01 · * p<.05 · ns = not significant` (FDR-corrected across classes).
"""),
cell("code", r"""
import sys
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
sys.path.insert(0, str(Path('..').resolve()))
from functions import lf_classify as C

def show(md):
    display(Markdown(md))

def show_run(task, variant, classifier):
    r = C.load_run(task, variant, classifier)
    if r is None:
        show(f'> _no run found for **{task} · {variant} · {classifier}** — run 320/330 first_')
        return
    o = r['metrics']['overall']
    p = o.get('permutation_p')
    ci = o.get('balanced_accuracy_ci', [float('nan'), float('nan')])
    show(f"### {task} · {C.VARIANT_LABELS[variant]} · {C.CLASSIFIER_LABELS[classifier]}\n"
         f"- balanced accuracy **{o['balanced_accuracy']:.3f}** "
         f"(chance {o['chance_level']:.3f}; 95% CI {ci[0]:.3f}–{ci[1]:.3f})\n"
         f"- macro-F1 {o['macro_f1']:.3f} · accuracy {o['accuracy']:.3f}"
         + (f" · **permutation p = {p:.4f}**" if p is not None else "")
         + f" · n={o['n_samples']} over {o['n_patients']} patients")
    for stem in ('confusion_matrix', 'per_class_strength', 'class_ersp_profile',
                 'class_feature_heatmap', 'coef_heatmap', 'permutation_null',
                 'feature_importance'):
        if stem in r['figures']:
            display(Image(filename=str(r['figures'][stem])))
    cols = [c for c in ['class', 'support', 'recall', 'recall_ci_lo', 'recall_ci_hi',
                        'precision', 'f1', 'roc_auc_ovr', 'perm_p_fdr']
            if c in r['per_class'].columns]
    display(r['per_class'][cols].round(3))
"""),
cell("markdown", r"""
## All experiments at a glance
One row per run. Sort/scan the `balanced_accuracy` vs `chance_level` gap and the
`permutation_p` column to see which representations decode best.
"""),
cell("code", r"""
runs = C.list_runs()
if not len(runs):
    show('> _no runs yet — execute 320 and 330 first._')
else:
    display(runs[['task', 'variant', 'classifier', 'balanced_accuracy', 'chance_level',
                  'macro_f1', 'permutation_p']].round(4))
    r2 = runs.copy()
    r2['exp'] = r2.task + ' · ' + r2.variant + ' · ' + r2.classifier
    r2 = r2.sort_values('balanced_accuracy')
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(r2) + 1))
    ax.barh(r2['exp'], r2['balanced_accuracy'], color='#4363d8', alpha=0.85)
    for i, (ba, ch) in enumerate(zip(r2.balanced_accuracy, r2.chance_level)):
        ax.plot([ch, ch], [i - 0.45, i + 0.45], color='#cc0033', lw=2)
    ax.set_xlabel('balanced accuracy   (red tick = chance)')
    ax.set_title('Separability across all classification experiments')
    plt.tight_layout(); plt.show()
"""),
cell("markdown", r"""
# Task A — Condition decoding (audio / picture / reading)

Chance = 0.333. A high diagonal here means the spectro-temporal response itself carries
*which modality* the electrode was engaged by. Watch the confusion matrix for the
audio↔reading vs picture structure, and the feature-importance panel for whether high-gamma
or lower bands carry the signal.
"""),
cell("code", r"""
for v in C.VARIANTS:
    for clf in ('logreg', 'rf'):
        show_run('condition', v, clf)
"""),
cell("markdown", r"""
# Task B — Parcellation decoding (Yeo networks)

Chance = 1/7 ≈ 0.143 (Yeo-7) and 1/17 ≈ 0.059 (Yeo-17). Because classes are imbalanced, read
**per-class strength** carefully: some networks (with many electrodes, broad coverage) will
separate strongly while sparse networks stay at chance. The feature-importance **by-condition**
panel answers "does one condition carry most of the anatomical signal, or all three jointly?".
"""),
cell("code", r"""
for n_net in (7, 17):
    show(f'## Yeo-{n_net}')
    for v in C.VARIANTS:
        for clf in ('logreg', 'rf'):
            show_run(f'parcellation_yeo{n_net}', v, clf)
"""),
cell("markdown", r"""
# Takeaways & caveats

- **Above-chance ≠ strong.** A run can clear the permutation null yet still confuse most
  classes — always cross-check balanced accuracy, the confusion matrix, and the per-class
  stars together.
- **Per-class is where the story is.** The summary number averages over classes; the
  per-class strength panel shows *which* conditions/networks are actually recoverable and
  which collapse into their neighbours.
- **Patient-grouped CV is conservative.** Numbers are lower than a naive random split would
  give, but they are the honest estimate of decoding in an unseen patient.
- **Feature importance is associational**, not causal — it tells you which bands/time/condition
  the model leaned on, given everything else it had.

_To refresh this page: re-run 320 / 330 (writes new runs), then re-run this notebook — it
always loads the latest run per (task · variant · classifier)._
"""),
]

write_nb("310_classification_dataprep.ipynb", nb310)
write_nb("320_condition_classification.ipynb", nb320)
write_nb("330_parcellation_classification.ipynb", nb330)
write_nb("390_results.ipynb", nb390)
print("all notebooks written.")
