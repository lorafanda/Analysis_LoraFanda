# Runbook — a new cohort, end to end

What to run, in order, after any change that touches the ERSP cubes: a bad-channel
addition, a white-matter reference change, a 140 re-run, a patient arriving or leaving.
Written for v16 (2026-10-12); the only thing that changes next time is the version number.

Two machines, and the split is not optional:

* **server** has the data and the environments, no site checkout and no git remote.
* **laptop** has the site checkout and the remote. Every site write and every push
  happens here.

Set the interpreter once per shell. On the laptop:

```bash
PY="C:/Users/fanda/AppData/Local/Programs/Python/Python311/python.exe"
export PYTHONIOENCODING=utf-8
```

On the server use the LORA_ENV2 interpreter instead, and never pip-install anything from
a notebook while a kernel is alive.

---

## Stage 0 — the cubes must be true before anything reads them (server)

Nothing downstream can detect a stale cube. `rebuild_concat_cache` decides a cache hit on
parameters only — no patient list, no timestamps, no content hash — so a leftover file
from an older run enters the cohort looking current. v8 carried 48 of them.

```bash
$PY 01_FBM_Analysis/141_audit_140.py
```

One row per patient: what 140 was run with and what came out. Config is read live, so the
bad-list and reference columns say what the code would do **now** — a patient whose row
disagrees with its own run needs re-running before you go on.

```bash
$PY 01_FBM_Analysis/142_stale_cubes.py
```

Lists, per patient, every file older than that patient's own 140 run — exactly the
channels a re-run no longer produces: newly bad-listed, newly a WM reference, newly
Unknown. Read the list, then:

```bash
$PY 01_FBM_Analysis/142_stale_cubes.py --delete
```

Re-run 140 for any patient 141 flags, one patient per terminal, three in parallel is fine:

```bash
$PY 01_FBM_Analysis/140_ersp_pipeline.py --patient PAT_3455
```

Use the id **config** spells, not the output folder: MicroEPI patients are `G-01` … `G-06`,
and `--patient PAT_6854` sends 140 looking for HUG sEEG data that does not exist.

---

## Stage 1 — the cohort cache (server)

```bash
cd 02_FBM_Clustering
$PY rebuild_concat_cache.py --dry-run --version 16
```

Read the dry run before committing to it. It names the patients entering and leaving, and
the ones present on disk but excluded by `lf_concat.DEFAULT_EXCLUDE_PATIENTS`.

```bash
$PY rebuild_concat_cache.py --apply --version 16
```

A **new directory** every time (`outputs/_dataset/concat_source_v16`), never a rebuild in
place — that is the only thing that forces the walk of `ERSP_matrix` and so the only thing
that recomputes `high_activity`. Check the three lines it prints: GATED cohort, X_concat
shape, and the per-patient table with the deltas against v15.

`lf_concat.DEFAULT_CONCAT_CACHE` picks the **highest** `concat_source_v<N>` on disk by
itself, so there is nothing to edit — unless `LF_CONCAT_CACHE` is set in the environment,
which overrides it. It is currently unset; keep it that way.

FIG C.0 is rebuilt automatically at the end of this step. If it reports a non-zero exit,
run `$PY make_cohort_figure.py` by hand; the cache itself is fine.

---

## Stage 2 — the fits (server)

Every run on disk was fitted to the previous cohort and is now stale. These three read the
cache read-only and must never rebuild it. They are independent — three terminals.

```bash
$PY 240_cluster_kmeans.py
$PY 241_cluster_hierarchical.py
$PY 242_cluster_cnmf.py
```

Each fits K = 5…30 over all four feature sets (`concat_hg`, `concat_rawds`,
`concat_bands5`, `concat_bands5z`) and then the bi-cross-validated held-out curve. To cut
it down:

```bash
$PY 240_cluster_kmeans.py --ks 8 --feature-set concat_bands5z --no-heldout
```

Then K, with the three criteria side by side rather than the one that cannot work:

```bash
$PY choose_k.py
```

Remember what the held-out peak does to patient generalisation: at that K, 15–45 % of
electrodes land in one-patient clusters. K = 8 is the clean choice.

`249_cluster_statistics.ipynb`, `250_recon_fsaverage.ipynb` and `252_clustering_recon.ipynb`
are still notebooks, not scripts. 249 is what produces the report cards, chips and
rasters; run it in Jupyter when you want those, and skip it otherwise.

---

## Stage 3 — the paper figures (server)

```bash
$PY 00_Paper2_Figures.py
```

Everything, every feature set. One at a time while arguing with a layout:

```bash
$PY 00_Paper2_Figures.py --figure 1 --feature-set concat_hg
$PY 00_paper2_figures2_2.py                  # FIG 2, agreement
$PY 00_paper2_figure3_lana.py                # FIG 3, LanA
$PY 00_paper2_figure4_correspondence.py      # FIG 4, the permutations (slow)
$PY 00_paper2_figure4_panels.py              # FIG 4, redraw from the tables (seconds)
$PY 00_paper2_figure5_confusion.py           # FIG 5
```

FIG 4 is split on purpose: `correspondence` does the permutation work and writes tables,
`panels` redraws from them, so iterate on the layout with `panels` alone.

---

## Stage 4 — the site and the visualizer (laptop only)

The page fetches every figure from `raw.githubusercontent.com/lorafanda/Analysis_LoraFanda/main`,
so a figure that is not pushed is a broken image, and a run folder is never on main.

Start by asking what is actually stale:

```bash
$PY 02_FBM_Clustering/check_site_ready.py --k 8
$PY 02_FBM_Clustering/audit_site_figures.py
```

`check_site_ready` walks the chain and marks each item OK / STALE / MISSING.
`audit_site_figures` goes further and names the script that regenerates each figure — use
its output as the task list rather than trusting this file to stay complete.

The visualizer's two data bundles, then the commits:

```bash
cd 02_FBM_Clustering
$PY make_coverage_bundle.py
$PY make_centroid_bundle.py
$PY scripts/push_cluster_bundle.py                    # plan only
$PY scripts/push_cluster_bundle.py --commit --push
```

That pushes in commits under the repo's guard (25 MB per commit, 5 MB per file) using
`git commit --only` with an explicit pathspec — never a bare `git commit`, because the
index holds thousands of files that are not yours to commit.

The site blocks, then the reading layer last:

```bash
$PY make_cohort_figure.py --insert
$PY make_paper_figures_webblock.py --insert
$PY make_outline_tab.py --insert
$PY make_site_ui.py --insert
```

`make_site_ui.py --insert` is idempotent and must come **after** any block insert: it is
the runtime layer (navigation, sub-tabs, lightbox, phone layout) and it wraps whatever is
on the page at the time.

Finally commit the page from the site checkout:

```bash
cd ~/lorafanda.github.io
git commit --only -F <message file> -- analysis_status.html
git push
```

---

## The order, in one line each

1. `141_audit_140.py` — is any patient's run older than its config?
2. `142_stale_cubes.py [--delete]` — what a re-run left behind
3. `140_ersp_pipeline.py --patient <config id>` — for every patient 141 flagged
4. `rebuild_concat_cache.py --dry-run --version N`, then `--apply`
5. `240` / `241` / `242`, then `choose_k.py`
6. `00_Paper2_Figures.py` and the per-figure scripts
7. `check_site_ready.py`, `audit_site_figures.py` — what is stale
8. `make_coverage_bundle.py`, `make_centroid_bundle.py`, `scripts/push_cluster_bundle.py --commit --push`
9. the `--insert` generators, `make_site_ui.py --insert` last, then commit and push the page

Skipping a step does not fail loudly. That is the whole reason this file exists.
