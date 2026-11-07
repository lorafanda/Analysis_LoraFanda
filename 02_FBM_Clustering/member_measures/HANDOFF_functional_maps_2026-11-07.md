# Functional maps: handoff, 7 November 2026

What was done in the analysis session of 3–7 November, for another session to read. The site tab
"Functional maps · feature set & K" (analysis_status.html, section `#fmaps`) and
`functional_map_analysis_2026-11-05.docx` are both generated from the same code (see the end), and every number in them
is read from the CSVs below when the tab is built.

## The question
Which feature set × K × loading carries functional information against fMRI meta-analytic maps (NiMARE / NeuroSynth),
without leaning on clusters that are more than 50% one patient? Many-to-many fits are allowed: one cluster in several maps,
several clusters in one map. The main solution is 15 bands × convex NMF, K 7, cohort v17 (1587 electrodes, 29 patients).

## The maps (25)
`0_nimare_maps.py` → `04_FBM_Pooling/neurosynth_nimare/` (maps.csv, study_counts.csv). The 25 maps are:
- 10 single terms
- 7 task contrasts
- 8 task vs resting state: pictures, listening, sentences, naming, semantic, production, reading (592 studies), and heard
  sentences (sentence AND listening terms, 154 studies, the one map below the 200-study rule, kept by `MIN_OVERRIDE`)

The last two were added on 7 November, and so was the study-set syntax `("AND", list1, list2)`.
`4_neurosynth.py --ns nimare` samples every electrode (5-mm sphere, inside = the map's top 10%) → `out/nimare_z.csv`.
**Every step must get `--ns nimare`** (the default reads the old NeuroSynth website maps).

## One test everywhere (steps 13, 15, 16, 17)
- **Null:** within each patient, electrodes in shaft order (shaft, contact number, from the run's raw contact names), with
  the cluster labels rotated by a random offset, 2,000 times.
- **Test:** z = (share − mean of the shifts) / SD; p = 2 × normal tail; Benjamini–Hochberg over every bar of the figure
  (step 17: over the configuration).
- **Calls:** above chance = z > 0 and q < 0.05; below chance = z < 0 and q < 0.05.
- **Chance box:** the grey box = mean ± z_crit × SD, where z_crit is the |z| at the BH cut-off. A bar outside the box is
  exactly a significant one; step 16 prints a self-check, 0 exceptions every time.
- **Shared code:** `mm_common.shaft_z` (the test), `mm_common.bh`, `mm_common.save_styles` (the figure versions).
- **Same shifts:** each solution gets a fresh `default_rng(0)`, so the same bar has the same z in every step. This was
  checked for 15 bands K 7: steps 13 = 15 = 16 = 17, max |Δz| = 0. Step 13 originally used the normalised keys for the
  shaft order (different shifts, Δz up to 0.26); it now reads the run's raw names.
- **Figure look (`--v4` in 13, 15, 16):** full colour = above, pale = neither, white with a coloured edge + ▼ = below; dark
  line = chance level. No bootstrap whisker, no cohort tick.
- **Versions (`--styles screen poster paper`):** `_poster` is the full width at 72% of the height; `_paper` is 60% of the
  width, 51% of the height, text × 0.62, 600 dpi. Legends are rebuilt per size (`Legend._init_legend_box` + `set_loc`).

## Results (15 bands K 7, loading ≥ 0.35; `out/highlight_maps/15band_K7_load035_all_v4.csv`)
Cut-off |z| ≥ 2.58; 29 bars above, 15 below.

| Cluster | Above chance | Below chance |
|---|---|---|
| c1 (84) | 18 of 25 maps, at 1.8–3.1 × the share of all electrodes | none |
| c5 (132) | 7 sentence / semantic / listening maps | none |
| c0 (219) | 4 picture-naming / object maps | sentences vs words |
| c2, c6, c4, c3 | no map | at least one each (c4 in 9, incl. language vs rest z −4.2) |

The professor's four maps vs rest:
- heard sentences → c1 and c5 above
- pictures → c0 above
- reading → c1 above, c4 below
- production → c1 above, c4 below

No cluster is above chance in naming or pictures (the single terms).

**Step 17 (`out/maps_by_k/`):** where every cluster is multi-patient, no feature set, K (6–10) or loading (0 / 0.3 / 0.35 /
0.4) has more than 3 clusters above chance in ≥ 2 maps. 15 bands K 7 reaches 3 at loadings 0.3–0.4; so do 5 bands K 6–8
and 2 bands K 6–8. Accepting one-patient clusters, the most is 4 (5 bands K 10, loading 0.4, with 4 one-patient
clusters).

**Step 14 (`out/functional_information/`, 500 shifts):** the best valid configuration per K is
- K 6: 5 bands weighted z 16.6
- K 7: 2 bands weighted 17.6, then 15 bands 0.35 at 13.2 and 5 bands weighted at 12.7
- K 8: 2 bands weighted 24.2
- K 9: none valid
- K 10: 2 bands weighted 27.4

**Conclusion in the tab** (only claims the tables back):
- one language-network cluster
- two more specific functions
- the other four types avoid language maps
- the professor's four
- more clusters do not mean more functions
- 15 bands K 7 holds (function does not single it out; stability and agreement do)

## Files
| What | Where (02_FBM_Clustering/member_measures/) |
|---|---|
| per-cluster figure (tab section 4) | `out_k7/nimare/cluster_loadings/15band_cluster_loadings_load035_shaft_v4[_poster/_paper].png` + `.csv` |
| functional information (section 3) | `out/functional_information/functional_information[_poster/_paper].png` + `.csv` |
| higher K (section 5) | `out/maps_by_k/maps_by_k.csv`, `maps_by_k_best.csv` (17_maps_by_k.py, new 7 Nov) |
| cross-feature, K 6–10 (section 6) | `out/cross_feature_maps/K{k}_load035_v4[_poster/_paper].png` + `.csv` |
| highlight figures (section 7) | `out/highlight_maps/15band_K7_load035_{all_v4,sel_v4,v4}`, `15zband_K7_load035_v4`, `5band_K6_load035_v4` (+ `_noglass`, `_poster`, `_paper`) |
| tab + docx generator | `../make_functional_maps_tab.py` (`--docx` writes the Word document from the same HTML) |
| everything in one go | `update_functional_maps.py` |

The old figures (counting test, 23 maps) are still on disk without the `_v4` suffix. The tab no longer shows them.

## Later on 7 November
- **c5 renamed** from "production" to "response broadband increase", in `00_Paper2_Figures.TYPE_NAMES` (every figure that
  names types). The poster cluster blocks were redrawn with `make_poster_blocks.py --sets concat_bands15`.
- **The chosen-maps figure (`--maps`) has 6 maps:** production vs motor is out. With it, c3 loses its only significant
  mark in that figure; c3 is still below chance in production vs motor and naming in the 25-map figure.
- **The tab conclusion now opens with two points:**
  - Distinct algorithms find the same types: convex NMF / k-means / Ward, 7 of 7 matched pairs above both nulls, 52–73%
    of electrodes in the matched type against 14–15% chance.
  - The types need frequency-specific information that HFA alone loses: 15 bands vs 5 / 2 bands / HFA gives 60 / 40 / 21%
    against 14 ± 4%. HFA is not above chance (p = 0.059), and HFA alone passes the one-patient rule only at K 6.

  Both are read when the tab is built, from the newest `outputs/clustering/paper_figures/FIG4_K07_run*_caption.txt` and
  step 14's table.

## Update command
From `02_FBM_Clustering/member_measures/`, on the laptop with the desktop unlocked:

    python update_functional_maps.py

It reruns steps 13–17 with the flags above, rebuilds the tab and the docx, and splices them into the site
(`--tab-only` skips the steps; `--no-site` skips the site). Then commit with pathspecs (`git commit --only -- …`, as the
script prints) and push both repos.

## Open
- Steps 11 (composition) and 12 (best maps) still use the counting test; they are not on the tab.
- The ARI 0.15 between 2 bands and 15 bands in the short answer comes from the agreement analysis, not from these tables.

## Night of 7–8 November: poster figures
- **Step 16 `--v4`:**
  - no figure title; the website caption's bold first line carries it
  - smaller legend above the bars, in rows of three
  - centroids at 1 : 3 without y values
- **`make_poster_blocks.py`:**
  - left | top | right brains (`--views`)
  - centroid at 1 : 3 (`--aspect 3`) with no frequency values
  - the type name on one line
- **Frequency axis:** only the poster's example figure carries it.
- **`make_poster_assets.py`:**
  - The task screens are drawn by `task_header()` instead of cropped from the FIG0 PNG; the crop fractions had gone
    stale and cut the boxes.
  - `figure1_column.png` is now the task screens (no "stimulus" / "response" words) over the example electrode's
    15-band ERSP: three square panels, every band labelled, transparent background. The full ERSP and the cohort brains
    are removed.
- **Rename redrawn everywhere:** every figure that prints type names was redrawn with c5 = "response broadband
  increase": cluster cards, poster assets, member measures, FIG 6 (screen and paper), FIG 1 paper.
  `00_Paper2_Figures.py --figure 1` needs `--k 7` (its default is the held-out peak, K 30).
- **Commit limits:** the repo's pre-commit hook blocks a file > 5 MB or a commit > 25 MB. Split commits; never use
  `--no-verify`.
- **Maybe later:** transparent backgrounds for the cluster blocks, strips and step-16 figures (for a presentation).
