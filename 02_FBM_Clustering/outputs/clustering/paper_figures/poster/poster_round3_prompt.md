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

4. Row 4, "Where they sit": the Neurosynth box gets neurosynth_15band_enrichment.png, placed at the full width of its slot. Caption: "How often each type's electrodes fall inside each NeuroSynth association map, as enrichment against the cohort rate (1 = as often as any electrode); outlined cells have their whole patient-bootstrap interval above 1. The audio-stimulus type sits in the auditory, speech-perception and speech-production maps 6, 4 and 3 times more often than the cohort; production in the semantic map, low-frequency rise in retrieval, beta decrease in visual, each about 1.5 times."

## Files (under S:\HumanNeuronLab\ANALYSIS\FLM\Analysis_LoraFanda\02_FBM_Clustering\outputs\clustering\paper_figures\poster\)
- assets\figure1_column.png (2220 x 3060 px)
- assets\agreement_bars.png (bars only)
- assets\stability_row.png (consensus matrix and the K panel)
- assets\neurosynth_15band_enrichment.png (copied from member_measures\out\neurosynth_enrichment\15band_enrichment.png)
