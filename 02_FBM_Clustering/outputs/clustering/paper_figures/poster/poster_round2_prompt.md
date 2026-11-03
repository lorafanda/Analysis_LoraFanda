# Round 2 for Claude Design: paste below the line, attach the files named (2026-11-03)

---

Keep the A0 layout and the colour rules. Change these rows; every other box stays as it is.

Row 1, left box: remove the heading. Replace the text with: "Functional brain mapping before epilepsy surgery relies on electrical stimulation, which is slow and subjective. If task-evoked spectral responses recur across patients, they could map function without it. We ask whether 1587 electrodes in 29 patients fall into recurring response types, and whether those types survive a change of description, algorithm and patients."

Row 1, middle box "The task and the cohort": replace the figure with row1_task_cohort.png. It is one strip of five equal tiles (brain from the left, from above, from the right; the task screens; one electrode as the pipeline sees it). Place it at the full width of the box, tiles equal, nothing cropped. Keep the text under it, two lines shorter: "Three naming tasks, each a stimulus period and a spoken response. 29 patients, Geneva and Bern: 2917 contacts seen, 1587 kept by a response gate (at least 2 % of time-frequency bins above +2.2 dB)."

Row 2 "Seven response types": replace the strip with concat_bands15_grid.png: two rows of blocks, four and three, each block an ERSP of width-to-height 3:1 over its two brains. Keep the blocks at equal size; the empty slot bottom right stays empty. Replace the caption with one line: "The seven types at K 7: the mean ERSP of each type's members (±2 dB, audio | picture | sentences) and its electrodes on the glass brain, loading ≥ 0.3 drawn, size and colour by loading."

Row 3, left "Agreement": replace the figure with agreement_bars_table.png (grey bars above, the table below), full box width. Keep the text.

Row 3, middle "Stability": replace the figure with stability_row.png (the consensus matrix left, the K curve right), full box width. Replace the text with: "Refitting on 80 % of the patients 200 times returns the same partition: 0.23 ambiguous electrode pairs against 0.43 ± 0.03 for a Gaussian cloud with the data's covariance. Per type the consensus runs from 0.81 to 0.53. The held-out variance never peaks; K 7 is the last K before a cluster is over half one patient (36 % at K 7, 100 % at K 8), and 90 % of the K 7 partition survives at K 8."

Row 3, right "Separation": replace the figure with separation_row.png (one patient out, half the patients out, and the silhouette of every electrode), full box width. Keep the text.

Row 4: swap the proportions to 1:2, "Where each type stands" on the left at one third, "Where they sit" on the right at two thirds. In "Where they sit" leave a placeholder box of the same height to the right of the atlas panel, titled "Neurosynth", empty for now.

## Files (all under S:\HumanNeuronLab\ANALYSIS\FLM\Analysis_LoraFanda\02_FBM_Clustering\outputs\clustering\paper_figures\poster\)
- assets\row1_task_cohort.png (4500 x 1065 px)
- concat_bands15_grid.png (4320 x 1990 px); concat_bands15z_grid.png if the companion strip is used
- assets\agreement_bars_table.png (2158 x 1568 px)
- assets\stability_row.png (3202 x 1465 px); the two halves singly as stability_consensus.png and stability_K.png
- assets\separation_row.png (3960 x 1290 px); assets\separation_violin_loading.png is the loading alternative to the silhouette panel
