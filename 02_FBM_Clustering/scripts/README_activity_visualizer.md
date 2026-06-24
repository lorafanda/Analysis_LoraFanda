# Activity Visualizer — interactive ERSP-on-brain page

A local, browser-based replacement for notebook 241's pre-rendered PNG/MP4
brain movies. The whole page is the fsaverage brain (Niivue/WebGL); a hideable
control bar lets you:

- **▶ Play** a sliding time window that sweeps across the trial (−1 → 7 s),
- pick **which frequency bands to average** (the 8 standard bands, multiselect),
- switch **condition** (all / picture / audio / reading),
- scrub time, change window width, playback speed, colour scale, and toggle
  pial ↔ inflated surface.

Press **`h`** to hide all panels (pure brain), **space** to play/pause.

## How it works (why it's instant)

The surface projection from notebook 241 is a *linear* operator on contact
values, and its Gaussian weights depend only on geometry. So instead of
re-rendering per (band, window, condition), we precompute **once**:

- a tiny `[condition × band × contact × time]` activity cube, and
- the vertex→contact projection weights (geometry only),

and the browser does band-averaging + time-window averaging + projection +
density-alpha live. Averaging a band subset / time window in the browser is
mathematically identical to re-running 241 for that exact window (exact for the
`mean` contact aggregation 241 uses).

## Files

| File | What |
|---|---|
| `precompute_activity_cube.py` | Builds the data bundle (run once, or when the 230/240 run changes). |
| `activity_visualizer.html` | The page. Vanilla JS + Niivue from CDN. |
| `outputs/250_recon/fsaverage/activity_viz/` | The generated bundle (manifest + binaries, ~14 MB). |

Data source (all inside `Analysis_LoraFanda` — `Analysis_Lora` is never used):
- **Activity**: the raw *ungated* ERSP dataset
  `04_FBM_Pooling/outputs/_dataset/pooling/_raw_ungated/X_3d.npy` (8841 samples ×
  129 freq × 300 time) + row-aligned `df_meta.parquet`. Built from
  `01_FBM_Analysis/outputs/04_ersp_LM_RAWONLY`.
- **Coords**: `250_recon/fsaverage/coords/ALL_PATIENTS_contacts_fsaverage.csv`
  (all cohorts, HUG + EL).
- **2901** contacts placed (cortical, ≤12 mm from pial). Electrode matching
  normalizes labels (`A_L10` ⇒ `AL10`, case-insensitive) so the underscore/case
  differences between the ERSP meta and the coords table line up.

Change the source by editing `DATASET_DIR` at the top of the script (e.g. point
it at a gated dataset instead of `_raw_ungated`).

## Running it

A local web server is required (browsers block `fetch()` of local files):

```powershell
cd S:\HumanNeuronLab\ANALYSIS\FLM\Analysis_LoraFanda\02_FBM_Clustering
# use the env that has numpy/pandas/pyarrow (here: Python 3.11)
& "C:/Users/fanda/AppData/Local/Programs/Python/Python311/python.exe" -m http.server 8000
```

Then open <http://localhost:8000/scripts/activity_visualizer.html>.

To regenerate the bundle (after a new clustering/recon run, edit the run IDs at
the top of the script first):

```powershell
& "C:/Users/fanda/AppData/Local/Programs/Python/Python311/python.exe" `
    02_FBM_Clustering/scripts/precompute_activity_cube.py
```

## Notes / gotchas

- Uses the `.gii` fsaverage meshes, **not** `.mz3`: the `.mz3` written by
  `build_fsaverage_meshes.py` carry a wrong magic word (`0x6D23` instead of the
  `0x5A4D` Niivue expects) and are rejected as "Invalid MZ3". The `.gii` share
  the exact same vertex order, so they stay aligned with the projection arrays.
  (This `.mz3` bug also affects the MOBA viewer — worth fixing at the source.)
- This page is **not** committed to the analysis repo (`*.html` is gitignored —
  web pages live in the `lorafanda.github.io` repo). The data bundle under
  `outputs/250_recon/fsaverage/activity_viz/` can be committed if you later want
  to deploy the page to the website.
