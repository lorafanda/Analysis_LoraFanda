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

**Two resolutions (`USE_DS`).** Default `USE_DS=True` explores the fast **15×30 band-downsampled**
grid via per-condition mean band×time heatmaps; `USE_DS=False` runs the full-res **129×300** blob
overlays, thinned by a tunable **`SCORE_PCT`** blob-score gate so they stay readable. See §1.

**The goal:** read these, then in the final cell type the boxcar + Gaussian windows for the
three zones — **perception**, **pre_articulation**, **audio** — saved to
`outputs/pooling/window_config.json` for `420` to pool over.
"""),
cell("code", SETUP),
cell("markdown", r"""
## 1 — Load the canonical dataset (+ choose resolution)
Ungated: every electrode × condition ERSP for patients with all three conditions (windowed
gating happens in `420`). The heavy walk is cached.

**Resolution toggle.** `USE_DS=True` works on the fast **15×30 band-downsampled** grid (the
clustering "rawds" rep — good for preliminary iteration); set it `False` for the full-res
**129×300** ERSPs (the real run). The blob overlays only make sense at full res, so in ds mode
they're replaced by per-condition mean band×time **heatmaps** (§2) and a power time-marginal (§3).

**Score gate (full-res only).** When `USE_DS=False`, `SCORE_PCT` drops the weakest blobs from the
overlays so they aren't an unreadable mess — slide it up to show only stronger blobs.
"""),
cell("code", r"""
# ---- knobs ----
USE_DS       = True     # True: fast 15x30 band-downsampled grid · False: full-res 129x300
SCORE_PCT    = 33.0     # full-res overlays only: drop blobs below this score percentile
DS_TIME_BINS = 30

df_meta, X_full = P.prepare_pooling_dataset(INPUT_DIR)
X = P.downsample_dataset(X_full, time_bins=DS_TIME_BINS) if USE_DS else X_full
score_min = None if USE_DS else P.resolve_score_gate(X_full, pct=SCORE_PCT)
print('samples:', len(df_meta), '| grid:', 'ds' if USE_DS else 'full', '| X:', X.shape)
df_meta.head()
"""),
cell("markdown", r"""
## 2 — Activity per condition
**ds mode** (`USE_DS=True`): per-condition **mean band×time heatmap** — bright patches = where
the dataset reliably responds. **full-res mode**: blob **overlays**, red = positive / blue =
negative, outline shade ∝ |mean dB|, thinned by the `SCORE_PCT` gate. Dense bands → candidate
pooling zones. (⚠️ overlay ellipses are moment-matched approximations of each blob's extent.)
"""),
cell("code", r"""
disc = P.new_run_dir('discovery')
print('discovery run:', disc)
for cond in P.CONDITIONS:
    if USE_DS:
        png = P.plot_ds_heatmap(X, df_meta, cond, disc / f'ds_heatmap_{cond}.png')
    else:
        png = P.plot_blob_overlay(X_full, df_meta, cond, disc / f'overlay_{cond}.png',
                                  score_min=score_min)
    display(Image(filename=str(png)))
"""),
cell("markdown", r"""
## 3 — Time-marginal
The quantitative read-off for choosing windows. **ds mode**: mean positive / negative band power
across contacts × bands at each time. **full-res**: count of contacts with a positive/negative
blob active at each bin. Peaks/plateaus = the windows worth pooling.
"""),
cell("code", r"""
for cond in P.CONDITIONS:
    if USE_DS:
        png = P.plot_ds_time_marginal(X, df_meta, cond, disc / f'ds_time_marginal_{cond}.png')
    else:
        png = P.plot_time_marginal(X_full, df_meta, cond, disc / f'time_marginal_{cond}.png',
                                   score_min=score_min)
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
USE_DS        = True                 # True: fast 15x30 ds grid · False: full-res 129x300
GRID          = 'ds' if USE_DS else 'full'
DS_TIME_BINS  = 30
FEATURE_SETS  = P.FEATURE_SETS       # ('hg', 'bands15')
WINDOW_SHAPES = P.WINDOW_SHAPES      # ('boxcar', 'gaussian')
N_PERM        = 0                    # >0 adds the circular-shift temporal-null p (slow)
SEED          = 42
print('grid:', GRID, '| zones:', list(cfg['zones']), '| feature sets:', FEATURE_SETS,
      '| shapes:', WINDOW_SHAPES, '| n_perm:', N_PERM)
"""),
cell("code", r"""
df_meta, X_full = P.prepare_pooling_dataset(INPUT_DIR)
X = P.downsample_dataset(X_full, time_bins=DS_TIME_BINS) if USE_DS else X_full
print('samples:', len(df_meta), '| grid:', GRID, '| X:', X.shape)
"""),
cell("markdown", r"""
## Pool & gate (the heavy step — cached to `outputs/_dataset/pooling/pool_table_<grid>.parquet`)
One row per contact × condition × zone × window shape × feature. Re-run only when the windows
or the upstream ERSPs change.

> ⚠️ **ds caveat:** on the 15×30 grid the σ/proportion qualification gate runs on the *smoothed*
> band-mean map — a coarse proxy for the full-res clustering gate. Use `USE_DS=False` for the
> final qualification numbers; `ds` is for fast preliminary exploration.
"""),
cell("code", r"""
df_pool = P.build_pool_table(df_meta, X, cfg, grid=GRID,
                             feature_sets=FEATURE_SETS, window_shapes=WINDOW_SHAPES,
                             n_perm=N_PERM, seed=SEED)
print('pool table:', df_pool.shape, '| grid:', GRID)
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
USE_DS = True                              # must match what you ran in 420
GRID   = 'ds' if USE_DS else 'full'
df_pool = P.load_pool_table(grid=GRID)
aparc   = P.ensure_aparc_cache()          # builds once (MNE), then cached
coords  = P.load_coords()
print('grid:', GRID, '| pool rows:', len(df_pool), '| aparc contacts:', len(aparc),
      '| coords contacts:', len(coords))
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
USE_DS = True                       # must match what you ran in 420
GRID   = 'ds' if USE_DS else 'full'
try:
    df_pool = P.load_pool_table(grid=GRID)
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
cell("markdown", r"""
# Methods rationale & key references

**What this analysis is called.** Per contact we pool baseline-normalized **event-related
spectral perturbation (ERSP)** power within *a-priori, hypothesis-driven time windows* — a
**confirmatory time–frequency region-of-interest (ROI) analysis**, the counterpart to the
data-driven **cluster-based permutation test** (Maris & Oostenveld 2007). Thresholding each
contact's windowed power against baseline to label it "responsive" is standard intracranial-EEG
practice; high-gamma (70–150 Hz) is the canonical task-response marker
(see also ERD/ERS, Pfurtscheller & Lopes da Silva 1999).

**Why two window shapes.** Pooling power over a window is a weighted temporal average — i.e.
convolving the signal with a kernel, where the window is a *taper* / *apodization* function.
The **boxcar** (rectangular) window weights the zone equally — the straight hypothesis test —
but its hard edges cause spectral leakage / edge sensitivity. The **Gaussian** taper down-weights
the edges, suppressing edge artefacts and tolerating trial-to-trial latency jitter; close
**boxcar ≈ Gaussian** agreement is the robustness signal (Harris 1978). In machine-learning terms
this is **temporal average pooling** (uniform vs Gaussian pooling kernel); the principled
generalization is **multitaper** estimation (Slepian/DPSS tapers; Thomson 1982).

**References**
- Pfurtscheller G & Lopes da Silva FH (1999). Event-related EEG/MEG synchronization and
  desynchronization: basic principles. *Clin. Neurophysiol.* 110(11):1842–1857.
- Maris E & Oostenveld R (2007). Nonparametric statistical testing of EEG- and MEG-data.
  *J. Neurosci. Methods* 164(1):177–190.  *(the data-driven alternative to a-priori windows)*
- Harris FJ (1978). On the use of windows for harmonic analysis with the discrete Fourier
  transform. *Proc. IEEE* 66(1):51–83.  *(boxcar vs Gaussian vs other tapers)*
- Hamilton LS, Edwards E & Chang EF (2018). A spatial map of onset and sustained responses to
  speech in the human superior temporal gyrus. *Curr. Biol.* 28(12):1860–1871.
  *(per-electrode time-window response typing + anatomical mapping — closest iEEG analogue)*
- Forseth KJ et al. (2018). A lexical semantic hub for heteromodal naming in middle fusiform
  gyrus. *Brain* 141(7):2112–2126.  *(a-priori windows + electrode responsiveness)*

*Related terms:* ERD/ERS · high-gamma / high-frequency broadband (HFB) · windowed band-power
averaging · time–frequency ROI · tapering / apodization · matched filter · temporal pooling.
"""),
]


# ============================================================
# 440 — region × time cascade (thesis figure)
# ============================================================
nb440 = [
cell("markdown", r"""
# 440 — Region × time cascade (thesis figure)

The confirmatory result as a **publication figure**: for each anatomical region, the **mean
high-gamma time-course** across its responsive contacts, drawn as a band on a region × warped-time
heatmap. Rows are **sorted by peak latency**, so the **perception → pre-articulation → audio**
cascade reads top (early) → bottom (late). One panel per condition (audio / picture / reading),
with the a-priori zones shaded behind.

**Why this rather than a surface video:** every row is a *real average over that region's
contacts* — no spatial interpolation, no implied coverage. It shows *timing across regions* (what
the pooling analysis is actually about) in a way that holds up to review.

Run `420` first (it defines the responsive set). Default regions = **Yeo-7** (no MNE needed);
switch `REGION_SCHEME` to `'yeo17'` or `'aparc'` (the latter reuses the aparc cache from `430`).
"""),
cell("code", SETUP),
cell("code", r"""
# ---- knobs ----
USE_DS          = True                # must match what you ran in 420
GRID            = 'ds' if USE_DS else 'full'
DS_TIME_BINS    = 30
REGION_SCHEME   = 'yeo7'              # 'yeo7' | 'yeo17' | 'aparc' (aparc needs 430's cache)
MIN_CONTACTS    = 5                   # drop regions with fewer responsive contacts
RESPONSIVE_ONLY = True                # restrict to contacts qualifying in >=1 zone (from 420)
SORT_BY         = None                # None: sort rows by mean peak latency; or a condition name
FEATURE         = 'hg'
print('grid:', GRID, '| regions:', REGION_SCHEME, '| min_contacts:', MIN_CONTACTS,
      '| responsive_only:', RESPONSIVE_ONLY)
"""),
cell("markdown", r"""
## 1 — Load the pool table, anatomy and the ERSPs
"""),
cell("code", r"""
df_pool = P.load_pool_table(grid=GRID)
coords  = P.load_coords()
aparc   = P.ensure_aparc_cache() if REGION_SCHEME == 'aparc' else None
df_meta, X_full = P.prepare_pooling_dataset(INPUT_DIR)
X = P.downsample_dataset(X_full, time_bins=DS_TIME_BINS) if USE_DS else X_full
print('pool rows:', len(df_pool), '| samples:', len(df_meta), '| X:', X.shape)
"""),
cell("markdown", r"""
## 2 — Build the cascade
Mean `FEATURE` time-course per region across its (responsive) contacts, one matrix per condition,
rows ordered by peak latency. The companion table lists each region's contact count and per-
condition peak latency (%).
"""),
cell("code", r"""
region_series = P.region_labels(df_meta, coords, aparc, scheme=REGION_SCHEME)
qualifies = P.responsive_contacts(df_pool) if RESPONSIVE_ONLY else None
cascade = P.build_cascade(df_meta, X, region_series, grid=GRID, feature=FEATURE,
                          conditions=P.CONDITIONS, min_contacts=MIN_CONTACTS,
                          qualifies=qualifies, sort_by=SORT_BY)
print('regions kept:', len(cascade['regions']))
display(P.cascade_table(cascade))
"""),
cell("markdown", r"""
## 3 — The figure
RdBu_r centred at 0 (red = power increase, blue = decrease), dashed line = response onset (50%),
shaded bands = the a-priori zones. The top-to-bottom progression of the warm patches *is* the
cascade. Saved (PNG + CSV) under `outputs/pooling/cascade/<scheme>/runs/<id>/`.
"""),
cell("code", r"""
cfg = P.load_window_config(P.OUTPUTS_ROOT / 'window_config.json')
run_dir = P.new_run_dir('cascade', REGION_SCHEME)
png = P.plot_cascade(cascade, run_dir / f'cascade_{REGION_SCHEME}.png', cfg=cfg)
P.cascade_table(cascade).to_csv(run_dir / f'cascade_{REGION_SCHEME}.csv', index=False)
print('saved ->', run_dir)
display(Image(filename=str(png)))
"""),
]


# ============================================================
# 450 — predefined time-frequency ROI pooling (condition/cluster-blind)
# ============================================================
nb450 = [
cell("markdown", r"""
# 450 — Predefined time-frequency ROI pooling

The **cluster-blind, condition-blind** pooling path. A fixed, physiology-driven library of
time-frequency **ROIs** (`functions/roi_config.py`) is applied to **every** electrode × condition
ERSP — independent of the 02 clusters and of audio/picture/reading. Each ROI is a 2-D box
(`f_rows × t_bins`, 1-based inclusive) + a **sign** hypothesis, encoding a documented marker
(sensory-onset HGA, sustained processing, pre-response planning, motor execution, alpha/beta ERD,
theta retrieval, …). Pooling turns each ERSP into a small citation-anchored feature vector.

- **Pool** = mean dB in the box. **Qualify** = the box clears the clustering σ/proportion gate in
  the ROI's **own sign** (`pos`/`neg`/`both`).
- **Option A aggregation:** a contact **expresses** an ROI if it qualifies in **≥1 condition**.
- This is **additive** — the zone-based `410–490` path is untouched.

> ⚠️ Two predeterminant issues to reconcile (see the validation table in §1): the **ds vs full**
> Hz ranges disagree for `alpha_beta_suppression` and `broadband_activation_index`; and the
> `broadband_activation_index` box-mean is **not** Manning's broadband index (a slope/offset
> measure) — treat it cautiously or redefine it.
"""),
cell("code", SETUP),
cell("code", r"""
# ---- knobs ----
USE_DS       = True                  # True: 15x30 ds grid · False: full 129x300
GRID         = 'ds' if USE_DS else 'full'
DS_TIME_BINS = 30
print('grid:', GRID, '| n ROIs:', len(P.ROI_PARAMS[GRID]))
"""),
cell("markdown", r"""
## 1 — The ROI map (with legend) + consistency check
Each box labelled `(a) (b) (c) …`, coloured red = positive / blue = negative / purple = both.
Same-box opposite-sign ROIs overlap, so positive and negative signs are also drawn separately.
The validation table flags any ds↔full Hz / sign mismatches.
"""),
cell("code", r"""
import matplotlib.pyplot as plt
disc = P.new_run_dir('roi', GRID + '_map')
for g in ('ds', 'full'):
    P.plot_roi_map(grid=g, out_png=disc / f'roi_map_{g}.png'); plt.show()
# pos / neg separated (so the early-onset vs early-suppression twins don't overlap)
P.plot_roi_map(grid=GRID, signs=('pos', 'both')); plt.title('positive / both'); plt.show()
P.plot_roi_map(grid=GRID, signs=('neg',));         plt.title('negative');       plt.show()
display(P.roi_legend(grid=GRID))
print('\nds <-> full consistency (hz_mismatch / sign_mismatch = reconcile):')
display(P.validate_roi_config())
"""),
cell("markdown", r"""
## 2 — Pool every contact against every ROI
Condition-blind: one row per (contact, condition, ROI). Cached to
`outputs/_dataset/pooling/roi_table_<grid>.parquet`.
"""),
cell("code", r"""
df_meta, X_full = P.prepare_pooling_dataset(INPUT_DIR)
X = P.downsample_dataset(X_full, time_bins=DS_TIME_BINS) if USE_DS else X_full
df_roi = P.build_roi_table(df_meta, X, grid=GRID)
print('roi table:', df_roi.shape)
df_roi.head()
"""),
cell("markdown", r"""
## 3 — Per-ROI qualifier counts (option A)
Distinct contacts expressing each ROI in ≥1 condition.
"""),
cell("code", r"""
counts = P.roi_counts(df_roi)
display(counts)
ax = counts.set_index('roi_tag')['n_contacts'].plot.barh(
    figsize=(8, 0.4 * len(counts) + 1))
ax.set_xlabel('# contacts expressing'); ax.set_title('ROI expression (option A: ≥1 condition)')
ax.invert_yaxis(); plt.tight_layout(); plt.show()
"""),
cell("markdown", r"""
## 4 — Where do they map? (ROI × Yeo-7)
Crosstab of expressing contacts by anatomical network — the condition-blind "where each signature
lives" map. (Yeo from the coords CSVs; no MNE needed.)
"""),
cell("code", r"""
coords = P.load_coords()
ct = P.roi_region_crosstab(df_roi, coords, scheme='yeo7')
display(ct)
"""),
]


write_nb("410_zone_discovery.ipynb", nb410)
write_nb("420_pool_and_qualify.ipynb", nb420)
write_nb("430_anatomy_mapping.ipynb", nb430)
write_nb("440_cascade.ipynb", nb440)
write_nb("450_roi_pooling.ipynb", nb450)
write_nb("490_results.ipynb", nb490)
print("all notebooks written.")
