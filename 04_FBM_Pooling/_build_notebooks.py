#!/usr/bin/env python3
"""Generate the 04_FBM_Pooling notebooks as valid nbformat-v4 JSON.

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


# Shared setup cell (server UNC path first, local relative fallback — mirrors 02/03).
SETUP = r"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from IPython.display import display, Markdown, Image
sys.path.insert(0, str(Path('..').resolve()))
from functions import lf_pool as P

INPUT_DIR = Path(r'\\nasac-m2.unige.ch\m-HumanNeuronLab\ANALYSIS\FLM\Analysis_LoraFanda\01_FBM_Analysis\outputs\04_ersp_LM_RAWONLY')
if not INPUT_DIR.exists():
    INPUT_DIR = Path('../01_FBM_Analysis/outputs/04_ersp_LM_RAWONLY').resolve()

print('INPUT_DIR :', INPUT_DIR, '| exists:', INPUT_DIR.exists())
print('OUTPUTS   :', P.OUTPUTS_ROOT)
print('COORDS    :', P.COORDS_DIR, '| exists:', P.COORDS_DIR.exists())
print('conditions:', P.CONDITIONS, '| zones:', P.DEFAULT_ZONES)
print('feature sets:', P.FEATURE_SETS, '| window shapes:', P.WINDOW_SHAPES)
"""


# ============================================================
# 410 — zone discovery
# ============================================================
nb410 = [
cell("markdown", r"""
# 410 — Zone discovery (blob overlays → manual windows)

The first, **exploratory** step of the confirmatory pooling pipeline. It loads the **same
canonical samples the clustering pipeline uses** (one sample = the trial-averaged ERSP per
electrode × condition, shape 129 × 300), overlays every contact's segmented blobs so you can
*see* where activity concentrates in time, and ends with a cell where **you hand-define** the
pooling windows.

**The warped time axis.** Each ERSP is time-warped **50% stimulus / 50% response**, so the
vertical dashed line at bin **150 (50%)** marks **response onset**: left of it = sensing /
perception, right of it = response / production.

**What to look for.** Each blob is drawn as a **red (positive) / blue (negative) ellipse
outline**, its shade scaled by the blob's mean dB. Stacked across all contacts, dense bands of
outlines reveal the time regions where the dataset reliably responds — the candidate **pooling
zones**. The time-marginal plot underneath quantifies the same thing (contacts active per bin).

**The goal:** read these, then in the final cell type the boxcar + Gaussian windows for the
three zones — **perception**, **pre_articulation**, **audio** — saved to
`outputs/pooling/window_config.json` for `420` to pool over.
"""),
cell("code", SETUP),
cell("markdown", r"""
## 1 — Load the canonical dataset (identical to clustering)
Ungated: every electrode × condition ERSP for patients with all three conditions. Windowed
gating happens later in `420`, so we keep everything here. The heavy walk is cached.
"""),
cell("code", r"""
df_meta, X_3d = P.prepare_pooling_dataset(INPUT_DIR)
print('samples:', len(df_meta), '| X_3d:', X_3d.shape)
print('patients:', sorted(df_meta.patient_id.unique()))
df_meta.head()
"""),
cell("markdown", r"""
## 2 — Blob overlays per condition
Red = positive (activation), blue = negative (suppression); outline shade ∝ |mean dB|. Dense
vertical bands = candidate pooling zones. ⚠️ Blobs are arbitrary shapes — the ellipse is a
moment-matched approximation (centre = centroid, axes = 2·std), so treat it as a *summary*
of each blob's extent, not its exact outline. (Segmenting every contact is the slow step.)
"""),
cell("code", r"""
disc = P.new_run_dir('discovery')
print('discovery run:', disc)
for cond in P.CONDITIONS:
    png = P.plot_blob_overlay(X_3d, df_meta, cond, disc / f'overlay_{cond}.png')
    display(Image(filename=str(png)))
"""),
cell("markdown", r"""
## 3 — Time-marginal blob density
The quantitative read-off: how many contacts have a positive (up, red) or negative (down, blue)
blob active at each time bin. Peaks/plateaus here are exactly the windows worth pooling.
"""),
cell("code", r"""
for cond in P.CONDITIONS:
    png = P.plot_time_marginal(X_3d, df_meta, cond, disc / f'time_marginal_{cond}.png')
    display(Image(filename=str(png)))
"""),
cell("markdown", r"""
## 4 — Define the pooling windows  ✍️  (edit this cell)
Fill in the three zones from what you saw above. **All numbers are percentages of the 0–300
warped axis** (50% = response onset). Each zone needs **both** a `boxcar` (the primary, equal-
weight window — `t_lo_pct`/`t_hi_pct`) **and** a `gaussian` (the robustness window —
`center_pct`/`sigma_pct`). The seed values below are placeholders; overwrite them.

> The pooling is *permissive by design*: each zone targets activity that is **definitely
> present** in its window — it does not try to be exclusive about activity elsewhere.
"""),
cell("code", r"""
cfg = {
    "schema_version": 1,
    "n_time": P.N_TIME,
    "stim_frac": P.STIM_FRAC,           # 50% mark = response onset = bin 150
    "axis_units": "percent_of_300_warped_axis",
    "zones": {
        # zone               boxcar (equal weight)         gaussian (centre-weighted)
        "perception":       {"boxcar": {"t_lo_pct": 0,  "t_hi_pct": 50},
                             "gaussian": {"center_pct": 25, "sigma_pct": 10}},
        "pre_articulation": {"boxcar": {"t_lo_pct": 40, "t_hi_pct": 60},
                             "gaussian": {"center_pct": 50, "sigma_pct": 8}},
        "audio":            {"boxcar": {"t_lo_pct": 50, "t_hi_pct": 100},
                             "gaussian": {"center_pct": 75, "sigma_pct": 12}},
    },
}
P.validate_window_config(cfg)
cfgp = P.save_window_config(cfg, P.OUTPUTS_ROOT / 'window_config.json')
print('saved ->', cfgp)
P.plot_window_preview(cfg);
"""),
]


# ============================================================
# 420 — pool & qualify
# ============================================================
nb420 = [
cell("markdown", r"""
# 420 — Pool power & qualify contacts

Reads the windows from `410` and pools ERSP power inside each, then asks — per contact — whether
the response is **significant in the clustering sense, restricted to the window**.

**Pooling** = a **time-weighted average** of the contact's power over the window (boxcar = equal
weight inside the box; Gaussian = centre-weighted). Run for **two feature sets**, identically:
- **`hg`** — the single 70–150 Hz high-gamma line (the canonical iEEG task-response marker).
- **`bands15`** — the 15 frequency bands **separately**, on the native 300-bin time axis.

**Qualification** = clustering's high-activity gate (`prop(>2.2σ) ≥ 0.02` **OR**
`prop(<−3.0σ) ≥ 0.04`), computed **only over the window's time columns** — so it is *comparable*
to clustering but window-restricted. **Both signs are kept** (a contact can qualify as `+`
activation or `−` suppression). Gaussian gate support = centre ± 2σ (95% mass).

> ⚠️ *Comparable, not identical:* a contact that passes whole-ERSP clustering gating can fail a
> narrow window — that asymmetry is the confirmatory test, by design.

Optional secondary robustness: a per-contact **circular-time-shift null p** (`N_PERM > 0`).
"""),
cell("code", SETUP),
cell("code", r"""
# ---------------- config knobs ----------------
cfg = P.load_window_config(P.OUTPUTS_ROOT / 'window_config.json')
FEATURE_SETS  = P.FEATURE_SETS       # ('hg', 'bands15')
WINDOW_SHAPES = P.WINDOW_SHAPES      # ('boxcar', 'gaussian')
N_PERM        = 0                    # >0 adds the circular-shift temporal-null p (slow)
SEED          = 42
print('zones:', list(cfg['zones']), '| feature sets:', FEATURE_SETS,
      '| shapes:', WINDOW_SHAPES, '| n_perm:', N_PERM)
"""),
cell("code", r"""
df_meta, X_3d = P.prepare_pooling_dataset(INPUT_DIR)
print('samples:', len(df_meta), '| X_3d:', X_3d.shape)
"""),
cell("markdown", r"""
## Pool & gate (the heavy step — cached to `outputs/_dataset/pooling/pool_table.parquet`)
One row per contact × condition × zone × window shape × feature. Re-run only when the windows
or the upstream ERSPs change.
"""),
cell("code", r"""
df_pool = P.build_pool_table(df_meta, X_3d, cfg,
                             feature_sets=FEATURE_SETS, window_shapes=WINDOW_SHAPES,
                             n_perm=N_PERM, seed=SEED)
print('pool table:', df_pool.shape)
df_pool.head()
"""),
cell("markdown", r"""
## Qualifier summary
Distinct qualifying contacts per condition × zone × window shape × sign (the gate is
feature-independent, so features are collapsed first). Compare boxcar vs Gaussian counts —
close agreement = the zone is robust to the window choice.
"""),
cell("code", r"""
summ = P.qualifier_summary(df_pool)
display(summ)

import matplotlib.pyplot as plt
piv = summ.pivot_table(index=['condition', 'zone', 'sign'],
                       columns='window_shape', values='n_contacts', fill_value=0)
ax = piv.plot.barh(figsize=(9, 0.4 * len(piv) + 1))
ax.set_xlabel('# qualifying contacts'); ax.set_title('Qualifiers per zone (boxcar vs gaussian)')
plt.tight_layout(); plt.show()
"""),
]


# ============================================================
# 430 — anatomy mapping
# ============================================================
nb430 = [
cell("markdown", r"""
# 430 — Anatomy mapping (where do the qualifiers cluster?)

Takes the qualifying contacts from `420` and asks **where they sit** and **how tightly they
cluster** anatomically — comparing the **boxcar** vs **Gaussian** windows throughout.

Three anatomical frameworks:
- **Yeo-7 & Yeo-17** functional networks — precomputed in the coords CSVs (no extra cost).
- **Desikan-Killiany gyri** (`aparc`) — built once via MNE + fsaverage (`ensure_aparc_cache`).
- **fsaverage surface renders** — qualifying contacts drawn on the pial surface, red(+)/blue(−).

Summaries reuse the clustering anatomy engine: per-zone **purity / entropy** (how
region-coherent) and **spatial compactness** (mm spread, hemisphere-mirrored).

> ⚙️ **Requires `mne` + `pyvista` + the fsaverage surfaces** (server-side). The first aparc build
> downloads ~50 MB and takes ~30 s; afterwards it is cached. The render cell needs off-screen
> PyVista (set below, before importing pyvista).
"""),
cell("code", r"""
import os
os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
os.environ.setdefault('MPLBACKEND', 'Agg')
""" + SETUP),
cell("code", r"""
df_pool = P.load_pool_table()
aparc   = P.ensure_aparc_cache()          # builds once (MNE), then cached
coords  = P.load_coords()
print('pool rows:', len(df_pool), '| aparc contacts:', len(aparc), '| coords contacts:', len(coords))
"""),
cell("markdown", r"""
## 1 — Attach anatomy to the qualifiers
Join each qualifying contact (on `patient_id` + normalized contact name) to its Yeo-7, Yeo-17,
Desikan-Killiany gyrus and fsaverage xyz. Unmatched counts are reported (naming differs across
GVA/PAT vs BERN/EL cohorts).
"""),
cell("code", r"""
dfq = df_pool[df_pool.qualifies].copy()
df_q7  = P.attach_anatomy(dfq, coords, aparc, n_networks=7)
df_q17 = P.attach_anatomy(dfq, coords, aparc, n_networks=17)
df_q7[['patient_id', 'contact_norm', 'condition', 'zone', 'window_shape', 'sign',
       'yeo_label', 'aparc_label']].drop_duplicates().head(20)
"""),
cell("markdown", r"""
## 2 — Purity & spatial compactness per zone × window shape
Each **zone** is one "cluster". For both window shapes we score region purity/entropy and
spatial compactness, so boxcar vs Gaussian are directly comparable. (`N_PERM_ANAT > 0` adds the
entropy permutation p — is the zone more region-coherent than chance.)
"""),
cell("code", r"""
N_PERM_ANAT = 1000
for n_net, dfq_net in ((7, df_q7), (17, df_q17)):
    for shape in P.WINDOW_SHAPES:
        sub = dfq_net[dfq_net.window_shape == shape]
        run_dir = P.new_run_dir('anatomy', f'yeo{n_net}', shape)
        print(f'\n=== Yeo-{n_net} · {shape} -> {run_dir} ===')
        anat, comp, id2name = P.summarize_anatomy(sub, aparc, coords, run_dir,
                                                  by='zone', n_perm=N_PERM_ANAT)
        display(anat); display(comp)
"""),
cell("markdown", r"""
## 3 — fsaverage surface renders
Qualifying contacts of each condition × zone on the pial surface (red = +, blue = −), six views.
Uses the **boxcar** window by default — switch `RENDER_SHAPE` to compare.
"""),
cell("code", r"""
RENDER_SHAPE = 'boxcar'
ren = P.new_run_dir('anatomy', 'renders')
zones = list(df_q7.zone.dropna().unique())
for cond in P.CONDITIONS:
    for zone in zones:
        pngs = P.render_zone_brains(df_q7, ren, condition=cond, zone=zone,
                                    window_shape=RENDER_SHAPE)
        for p in pngs[:2]:           # show 2 views inline to keep the notebook light
            display(Image(filename=str(p)))
"""),
]


# ============================================================
# 490 — results narrative
# ============================================================
nb490 = [
cell("markdown", r"""
# 490 — Pooling results (read me)

The **results page** for `04_FBM_Pooling`. It computes nothing heavy — it reads the artifacts
`410`/`420`/`430` wrote and lays them out with the story and a guide to reading each figure. Run
`410 → 420 → 430` first, then run this top-to-bottom.

---

## What was asked
A **confirmatory** test, built on the same per-electrode ERSPs as clustering:

1. **Do contacts respond in a-priori time zones?** Pool ERSP power inside hand-defined windows —
   **perception**, **pre_articulation**, **audio** — and qualify each contact with a
   *window-restricted* version of clustering's high-activity gate.
2. **Where do the responsive contacts cluster?** Map qualifiers to Yeo-7/17 networks and
   Desikan-Killiany gyri, and summarize their anatomical purity + spatial compactness.

## Built to be robust — two window shapes
- **Boxcar** (primary): equal weight across the zone — the straight hypothesis test.
- **Gaussian** (robustness): centre-weighted, so it down-weights the edges and tolerates
  latency jitter. **If boxcar ≈ Gaussian** (qualifier counts, anatomy purity), the zone is real
  and not an artefact of exactly where you drew the window. **If they diverge**, the effect sits
  near a window edge — treat it with caution.

## Two feature sets
- **`hg`** — 70–150 Hz high-gamma line, the canonical task-response marker.
- **`bands15`** — each of the 15 bands separately, so band-specific zone effects are visible.

## How to read each output
| Output | What it is | How to read it |
|---|---|---|
| **Qualifier summary** | distinct qualifying contacts per condition × zone × shape × sign | the headline: which zones recruit many contacts, and whether `+`/`−` dominate |
| **Per-zone anatomy** | purity, entropy, top-3 regions per zone | high purity / low entropy = the zone's contacts share an anatomy |
| **Spatial compactness** | mm spread of each zone's contacts (hemisphere-mirrored) | small `distance_mm` = a tight anatomical cluster |
| **Surface renders** | contacts on fsaverage, red(+)/blue(−) | the visual "where" — look for a coherent patch per zone |

> ⚠️ *Comparable, not identical to clustering:* the gate uses the same σ thresholds but only over
> the window, so windowed qualifier counts are **lower** than whole-ERSP high-activity counts —
> that's expected, and is the point of a zone-restricted confirmatory test.
"""),
cell("code", SETUP),
cell("code", r"""
def show(md):
    display(Markdown(md))

runs = P.list_runs()
if not len(runs):
    show('> _no runs yet — execute 410 → 420 → 430 first._')
else:
    display(runs)
"""),
cell("markdown", r"""
## Qualifier summary (from the cached pool table)
"""),
cell("code", r"""
import matplotlib.pyplot as plt
try:
    df_pool = P.load_pool_table()
    summ = P.qualifier_summary(df_pool)
    display(summ)
    piv = summ.pivot_table(index=['condition', 'zone', 'sign'], columns='window_shape',
                           values='n_contacts', fill_value=0)
    ax = piv.plot.barh(figsize=(9, 0.4 * len(piv) + 1))
    ax.set_xlabel('# qualifying contacts')
    ax.set_title('Qualifiers per zone — boxcar vs gaussian (robustness)')
    plt.tight_layout(); plt.show()
except FileNotFoundError:
    show('> _no pool table yet — run 420._')
"""),
cell("markdown", r"""
## Boxcar vs Gaussian — anatomy robustness
Per-zone purity + compactness side by side for the two window shapes (Yeo-7 latest runs). Close
columns = the anatomical story does not depend on the exact window.
"""),
cell("code", r"""
rows = []
for shape in P.WINDOW_SHAPES:
    rd = P.latest_run('anatomy', 'yeo7', shape)
    if rd is None:
        continue
    anat = pd.read_csv(rd / 'per_cluster_anatomy.csv')
    comp = pd.read_csv(rd / 'per_cluster_spatial_compactness.csv')
    m = anat.merge(comp[['cluster_id', 'distance_mm', 'n_with_coords']], on='cluster_id')
    m['window_shape'] = shape
    rows.append(m)
if rows:
    allm = pd.concat(rows, ignore_index=True)
    cols = ['zone', 'window_shape', 'n_total', 'purity', 'entropy_bits',
            'top_region', 'top_proportion', 'distance_mm']
    cols = [c for c in cols if c in allm.columns]
    display(allm[cols].sort_values(['zone', 'window_shape']).reset_index(drop=True))
else:
    show('> _no anatomy runs yet — run 430._')
"""),
cell("markdown", r"""
## Surface renders (latest)
The most recent fsaverage renders, if `430` produced any.
"""),
cell("code", r"""
ren = P.latest_run('anatomy', 'renders')
if ren is None or not (ren / 'renders').exists():
    show('> _no renders yet — run 430 section 3._')
else:
    for png in sorted((ren / 'renders').rglob('*.png'))[:12]:
        show(f'**{png.relative_to(ren)}**')
        display(Image(filename=str(png)))
"""),
cell("markdown", r"""
# Takeaways & caveats
- **Qualifier counts** answer "does each zone recruit a population of contacts?"; **anatomy
  purity / compactness** answer "do they sit somewhere coherent?".
- **Boxcar vs Gaussian agreement is the robustness signal** — report both; flag any zone where
  they disagree as edge-sensitive.
- **Windowed ≠ whole-ERSP gating.** Lower counts here than clustering's high-activity totals are
  expected — the window is the hypothesis.
- **Sanity check:** `audio` qualifiers should be auditory-network / temporal-gyrus heavy;
  `perception` should be early-window heavy. If not, revisit the windows in `410`.

_To refresh: re-run 420 / 430 (writes new runs), then re-run this notebook — it always reads the
latest run per (stage · target · shape)._
"""),
]


write_nb("410_zone_discovery.ipynb", nb410)
write_nb("420_pool_and_qualify.ipynb", nb420)
write_nb("430_anatomy_mapping.ipynb", nb430)
write_nb("490_results.ipynb", nb490)
print("all notebooks written.")
