# Pipeline flowchart — full source material

Everything behind FIG E.9. The edges below are the dependencies the **code actually
has** — which file reads which artifact — verified against the scripts, not remembered.

Repo: `Analysis_LoraFanda`. Two stages: `01_FBM_Analysis` (signal → ERSP) and
`02_FBM_Clustering` (everything after). A separate repo `lorafanda.github.io` holds
`analysis_status.html` and `clustering_visualizer.html`.

---

## What the diagram has to communicate

Ranked. If a design choice trades one of these away, trade from the bottom.

1. **The order things must run in**, and where that order is forced by data rather than
   by convention.
2. **Two orderings that look like independence and are not.** Both cost real time in
   August 2026. These are the highest-value content in the whole figure.
3. **Two invisible failure modes** — statistics valid at one K only, and re-running a
   feature set superseding rather than updating its run.
4. **Committing is a pipeline step**, not housekeeping, because both web tools fetch
   over HTTP from the repository.
5. Which steps are one-time, which are per-cohort, which are per-run.
6. The artifact passed along each edge.

---

## Nodes

Grouped by stage. `TYPE` is `notebook` / `script` / `artifact` / `webtool`.
`CADENCE` is how often it actually gets run.

### Stage A — `01_FBM_Analysis` · signal to ERSP

| id | label | type | cadence | detail |
|---|---|---|---|---|
| A0 | raw sEEG + prep0 TSVs | artifact | — | the inputs |
| A1 | `adjust_fixation_cross_duration.py` | script | one-time per patient | `trial_end = next_onset − (1.8 − U(0,0.2))s`. Has `--restore` and a `.bak` guard; running it twice is refused, not silently doubled. Applied to PAT_3965 and EL033. |
| A2 | `150_ERSP_analysis_pipeline_noTwarping.ipynb` | notebook | per cohort change | resample to 1000 Hz, STFT `nperseg=128 / nfft=256 / noverlap=108`, baseline `(−0.6,−0.1)`, `fmax=500`, time-normalise to 300 bins with `proportions=(0.0,0.50,0.50)` so bin 150 is stimulus offset |
| A3 | `move_halves_out_of_ersp_matrix.py` | script | **every time A2 runs** | moves `_half1`/`_half2` out of `ERSP_matrix/` into `ERSP_halves/` |
| A4 | `outputs/04_ersp_LM_RAWONLY/<patient>/LM/ERSP_matrix/` | artifact | — | 9,342 full cubes (and 19,380 half files, which live elsewhere) |

### Stage B — the cohort

| id | label | type | cadence | detail |
|---|---|---|---|---|
| B1 | `rebuild_concat_cache.py --apply` | script | per cohort change | **serial, alone, and first.** Writes a NEW cache directory. Output: `outputs/_dataset/concat_source_v4/` → 1693 gated / 2959 ungated electrodes, 27 patients |

### Stage C — the fits

| id | label | type | cadence | detail |
|---|---|---|---|---|
| C1 | `240_cluster_kmeans.ipynb` | notebook | per cohort/feature-set | k-means, K = 5..30, every feature set |
| C2 | `241_cluster_hierarchical.ipynb` | notebook | per cohort/feature-set | Ward. Genuinely independent of C1 |
| C3 | `242_cluster_cnmf.ipynb` | notebook | per cohort/feature-set | `run_decomposition.py` → `publish_decomposition.py` → `sweep_decomposition.py` |
| C4 | `243_cluster_archetypes.ipynb` | notebook | optional | `run_archetypes.py`. A fourth track; does not gate stage D |
| C5 | cell 7 of 240 / 241 / 242 | script | with each fit | `make_heldout_variance.py --from-cache …` → `heldout_variance_<method>.csv`. Bi-cross-validated (row AND column folds) — the only curve that can turn over, and therefore the only one that can choose K |
| C6 | `sweep_stability.py --new-concat [--native]` | script | per run | → `stability_by_k.csv`; with `--native` the refit uses the run's OWN method → `stability_by_k_native.csv` |
| C7 | `outputs/clustering/<method>/<feature_set>/runs/<YYYYmmdd_HHMMSS>/` | artifact | — | the run directory everything downstream resolves |

Feature sets: `concat_hg` (900 features), `concat_rawds` (1350), `concat_bands5` (450),
`concat_bands5z` (450, z-scored per band).

### Stage D — `249_cluster_statistics.ipynb`

| id | label | type | detail |
|---|---|---|---|
| D1 | 249 §1 — merge the sweeps | notebook cell | → `heldout_variance_ALL.csv`, then the K where convex NMF's held-out curve PEAKS, per feature set → `peak_k.json` (concat_hg 11, concat_rawds 12) |
| D2 | 249 §2 — `make_cluster_statistics.py` | script | separation vs a matched null, anatomical coherence, leave-one-patient-out, agreement between methods. 50 nulls per cell. → `statistics/<fs>_K<k>/` and `cluster_statistics.json` into each run |
| D3 | 249 §3 — the figures | script | `make_cluster_figures.py` → C.3a/b/c, C.8a/b/c; `make_heldout_figure.py` → C.13. Cut at `peak_k`, in the visualizer's own colour palette |

### Stage E — per-run assets (what the report embeds)

| id | label | type | detail |
|---|---|---|---|
| E1 | `make_missing_centroids.py` | script | one centroid chip per cluster **per K (5..30)**. `concat_hg`: mean dB with ±1 SD. `concat_rawds`/`bands5`: averaged ERSP with a per-bin SD dot |
| E2 | `make_centroid_rasters.py` | script | the second view of each cluster. `concat_hg`: EVERY electrode as a row, sorted by membership strength. `rawds`: the same plane with a proper SD legend |
| E3 | `252_clustering_recon.ipynb` | notebook | per-cluster glassbrains, by condition and by patient |

### Stage F — recon and the bundle

| id | label | type | cadence | detail |
|---|---|---|---|---|
| F0 | `251_recon_shared_data.ipynb` | notebook | one-time per patient | fsaverage meshes + `ALL_PATIENTS_contacts_fsaverage.csv` |
| F1 | `make_coverage_bundle.py` | script | **after every new run** | **the visualizer's run list lives here.** Builds `coverage_viz/manifest.json` and the per-run arrays |
| F2 | `clustering_visualizer.html` + its exported report | webtool | — | reads the bundle, the run directories and the centroid chips — all over HTTP from the repository |

### Stage G — `analysis_status.html`

| id | label | type | detail |
|---|---|---|---|
| G1 | `make_cluster_webblock.py --insert` | script | the 02 tab's statistics block, generated from the files — the prose too, so a verdict cannot outlive the numbers |
| G2 | `make_s2_gallery.py --insert` | script | one example of every figure the stage makes; checks **git**, not the disk |
| G3 | `make_kiss_tab.py --insert` | script | the plain-words tab; also `make_bands_figure.py` and the other explainers |
| G4 | `git add · git commit · git push` | step | **not optional** — see the callouts |

---

## Edges

`SOLID` = hard dependency, the target reads what the source wrote.
`DASHED-WARN` = an ordering that looks like independence and is not. **Draw these so
they are impossible to miss.**

| from | to | kind | label |
|---|---|---|---|
| A0 | A1 | SOLID | |
| A1 | A2 | SOLID | corrected trial windows |
| A2 | A3 | SOLID | 9,342 cubes |
| A3 | A4 | **DASHED-WARN** | must run, or the cohort triples |
| A4 | B1 | SOLID | |
| B1 | C1 | SOLID | |
| B1 | C2 | SOLID | |
| C1 | C3 | **DASHED-WARN** | 242 reads 240's `X_train` |
| C3 | C4 | **DASHED-WARN** | 243 reads 240's `X_train` too |
| C1 | C5 | SOLID | |
| C3 | C6 | SOLID | |
| C5 | C7 | SOLID | |
| C7 | D1 | SOLID | |
| D1 | D2 | SOLID | peak K |
| D2 | D3 | SOLID | the statistics |
| C7 | E1, E2, E3 | SOLID | |
| F0 | F1 | SOLID | meshes + coords |
| E3 | F1 | SOLID | |
| C7 | F1 | SOLID | |
| F1 | F2 | SOLID | |
| D3 | G1, G2, G3 | SOLID | |
| G1/G2/G3 | G4 | SOLID | |
| F2 | G4 | **DASHED-WARN** | the visualizer needs the commit too |

---

## Callouts

Four. These are the content, not decoration — a version of this diagram without them is
just a list of filenames.

### CALLOUT 1 — beside edge A3→A4 · severity HIGH

> **Skip `move_halves_out_of_ersp_matrix.py` and the cohort triples.** The loader globs
> `ERSP_matrix` for cubes; if the split-half files are sitting there it finds three per
> electrode and believes all three. 9,342 real cubes became 19,380 rows and the
> responsiveness gate went from 35.1% to 96.9% — with nothing failing.

### CALLOUT 2 — beside edges C1→C3 and C3→C4 · severity HIGH

> **Convex NMF and archetypal analysis do not build their own features — they take
> `X_train` FROM THE K-MEANS RUN.** So for a NEW feature set, 242 and 243 cannot start
> until 240 has finished it. The three notebooks are only parallel for feature sets 240
> already has.

### CALLOUT 3 — under stage D · severity HIGH · two parts

> **VALID AT ONE K ONLY.** Every number in §2 is scored against a null *refitted at that
> K*, so it means nothing at another cut. `cluster_statistics.json` carries the K it was
> computed at and the report checks it before showing a row — which is why the statistics
> section goes quiet when you move the K control.
>
> **Re-running a feature set does NOT update its run — it SUPERSEDES it.** Run ids are
> timestamps and every resolver takes the newest. The old run keeps its statistics, its
> stability sweeps and its place in the coverage manifest, and nothing resolves to it any
> more. `SKIP_IF_RUN_EXISTS` in the notebooks and `--force` on `run_archetypes` exist for
> exactly this.

### CALLOUT 4 — at G4 · severity HIGH

> **The site and the visualizer fetch everything from the repository.** A figure, a CSV
> or a manifest that exists only on your disk is a 404 to both — which is what
> *"No metrics published for K=9"* and the gallery's broken boxes both were. Committing
> is a step in the pipeline, not tidying up afterwards.

---

## The short version

For the footer. Correct as written — this is the sequence to run when nothing has changed
but the cohort.

```
1.  adjust_fixation_cross_duration.py  →  150_ERSP…ipynb  →  move_halves_out_of_ersp_matrix.py
2.  python rebuild_concat_cache.py --apply                    (serial, alone, first)
3.  240 + 241 together;  then 242;  then 243                  (242/243 read 240's run —
                                                               not parallel for a NEW feature set)
4.  cell 7 of 240 / 241 / 242    then   python sweep_stability.py --new-concat --native
5.  249                                                       (merge → peak K → statistics → figures)
6.  make_missing_centroids.py · make_centroid_rasters.py · 252  →  make_coverage_bundle.py
7.  make_cluster_webblock / make_s2_gallery / make_kiss_tab --insert  →  commit and push BOTH repos
```

---

## Design notes

What went wrong in the matplotlib version, so it is not repeated:

- **Seven stacked horizontal bands made it tall and thin** (20 × 28 in). At readable text
  size it does not fit a screen or a page. A denser arrangement — columns, or a
  left-to-right spine with branches — would suit the content better.
- **The two DASHED-WARN edges are the most valuable content and read as the least
  important**, because they are thin dashed lines among many arrows.
- Long arrows crossing several bands (C7 → F1, D3 → G1/G2/G3) tangle. Consider repeating
  a small artifact node near its consumers instead of drawing the full edge.
- Node captions run 2–4 lines. Either give them room by construction or move the detail
  to a numbered key beside the diagram.

Palette actually used (house colours, reuse or discard freely):

| role | hex |
|---|---|
| ink | `#1b232c` |
| muted | `#68727d` |
| grey / rules | `#c9ced4` |
| warning, severity | `#c1121f` |
| stage 01 | `#2c7fb8` |
| cohort | `#8a6d3b` |
| fits | `#41ab5d` |
| statistics | `#5b2c83` |
| per-run assets | `#e08214` |
| recon / bundle | `#0b7a75` |
| website | `#c0392b` |

Audience: the person rebuilding this pipeline — most often the author, months later.
It is a working reference to be followed step by step, not a diagram for a paper.
