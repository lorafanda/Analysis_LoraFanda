# Round 3 for Claude Design: paste below the line, attach the files named (2026-11-03)

---

Keep the A0 layout and the colour rules. Four changes.

1. Row 1, "The task and the cohort": replace the five-tile strip with figure1_column.png, one column: the task screens on top, under them one electrode's full ERSP and its 15-band summary aligned condition by condition, the three cohort brains at the bottom. Give the box the full height it needs; the text goes under the brains.

2. Row 3, "Agreement": keep only the bar figure (agreement_bars.png). Under it build a native table, six rows, five columns, with these values, in the body type size minus 4 pt, rules in #c9ced4 only, no fills, no bold except the header:

| comparison | diagonal share | random pairings | ARI | matched r |
|---|---|---|---|---|
| convex NMF vs k-means | 0.73 | 0.14 ± 0.11 | 0.51 | +0.95 |
| convex NMF vs Ward | 0.52 | 0.15 ± 0.08 | 0.26 | +0.78 |
| k-means vs Ward | 0.61 | 0.14 ± 0.09 | 0.33 | +0.81 |
| 15 bands vs 5 bands | 0.60 | 0.14 ± 0.09 | 0.38 | +0.77 |
| 15 bands vs 2 bands | 0.40 | 0.14 ± 0.06 | 0.15 | +0.57 |
| 15 bands vs HFA | 0.21 | 0.14 ± 0.04 | 0.07 | +0.39 |

Caption under the table: "Diagonal share: the fraction of electrodes on the matched diagonal of the two solutions' contingency table, against random pairings of the same two clusterings (1000 draws). Matched r: the centroid correlation of the matched pairs."

3. Row 3, "Stability": replace the figure with the new stability_row.png. The right panel now shows, in red, the number of clusters that are more than half one patient at each K (0 up to K 7, 1 at K 8), against the held-out variance in black. Text stays.

4. Row 4, "Where they sit": the Neurosynth box gets neurosynth_15band.png at the full width of its slot; if the box has the height, neurosynth_15zband.png under it at the same width, otherwise the 15-band panel alone. Replace the box text with: "Against the language atlas (LanA) no type is placed by the network beyond one modest effect each for the production type (share inside, shaft-shift null q 0.04) and the audio-stimulus type (left-hemisphere mean, 5000 spun maps, q 0.03). Against eight NeuroSynth maps one type has a home: the audio-stimulus type sits in the speech-perception and speech-production maps 4 and 3 times as often as the cohort, the low-frequency rise in the retrieval map 1.4 times; the other five sit where the cohort sits. With z-scored bands the auditory high-gamma type is sharper, 38 % of its electrodes in each speech map, and four of six clusters find a map. The types are task phases, not atlas regions." Caption under the panel: "NeuroSynth: the share of each type's electrodes inside eight association maps, one or two per task stage; the number is the share, the colour the share against the cohort's (log scale). Outlined: above the cohort in 97.5 % of patient resamples; the right column lists those maps. The z-scored companion is the same matrix with each cluster in the row of the published type it matches; the greyed cluster is 64 % one patient. No spatial null: neighbouring contacts fall in the same map."

## Files (under S:\HumanNeuronLab\ANALYSIS\FLM\Analysis_LoraFanda\02_FBM_Clustering\outputs\clustering\paper_figures\poster\)
- assets\figure1_column.png (2220 x 3060 px)
- assets\agreement_bars.png (bars only)
- assets\stability_row.png (consensus matrix and the K panel)
- assets\neurosynth_15band.png (2107 x 1452 px, the eight-map NeuroSynth label matrix, 15 bands, each map on the pial surface under its name)
- assets\neurosynth_15zband.png (2356 x 1452 px, the same for the z-scored companion)
