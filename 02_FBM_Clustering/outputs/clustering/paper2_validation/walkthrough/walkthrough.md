# The cluster visualizer, claim by claim

*Built 2026-11-01 16:06 from the files the page reads (cohort concat_source_v17, run 20261017_194301). Page: https://lorafanda.github.io/cluster_visualizer.html. The brain images are rendered from the same data with the page's own encodings on FIG 1's glass scenes; the click path under each one reproduces it live.*

## The request, restated

For each claim the paper makes about the clustering (the types, their agreement across descriptions and algorithms, their stability across patients, their separability) state the claim, say which view of the cluster visualizer shows it and which clicks get there, show the brain and the statistics that appear, and read the numbers off them with the paper each measure rests on. In short: claim → where to click → what you see → what it means.

## How to read a click path

`Run` is the feature set × algorithm matrix at the top of the control panel; `K` the row of numbers under Clusters; the chips `#1 … #7` are the clusters in figure order (#1 = the leftmost block of FIG 1); `View` has one / overlay / grid / held-out. In the held-out view, the panel on the right is clickable: a type bar, a matrix cell, a patient.

### The page itself, as it appears live

Screenshots of the live page at the time of building (800 px wide, the browser pane's size); the claims below use renders of the same states at print resolution.

![00 control panel k7 live](shots/00_control_panel_k7_live.jpg)

![01 live one view c5 K7](shots/01_live_one_view_c5_K7.jpg)

![02 live grid all seven types](shots/02_live_grid_all_seven_types.jpg)

![03 live grid position1 three algorithms](shots/03_live_grid_position1_three_algorithms.jpg)

![04 live heldout view](shots/04_live_heldout_view.jpg)


## Claim 1 · The clustering gives seven recurring response types, each carried by many patients

**Where to click.** Run `15 bands × cNMF` → K `7` → View `one` → chip `#1` (c5 production), then `#7` (c3). For all at once: View `overlay` and click every chip.

![c5](renders/01_type_c5.png)

![c3](renders/02_type_c3.png)

![all types](renders/03_all_types.png)

**What you see.** Every dot is one electrode of the type; its size and its fade carry its loading on the type (small and grey below 0.5), the chip carries n and the biggest patient's share, and the red * that marks a patient holding more than half of a cluster is absent at K 7.

**The numbers.** c5 production n 200 (17 patients, largest 17 %), c0 beta decrease n 318 (22 patients, largest 25 %), c2 broadband decrease n 168 (11 patients, largest 36 %), c6 response-period low-frequency fall n 224 (24 patients, largest 25 %), c4 low-frequency rise n 334 (28 patients, largest 12 %), c1 audio-stimulus low-frequency n 143 (22 patients, largest 20 %), c3 sentence-stimulus low-frequency n 200 (25 patients, largest 18 %); median own-type loading 0.41, 16 % below the 0.3 draw cut (Ding et al. 2010).

**What it means.** Seven shapes, each spread over 11 to 28 patients; the mixed loadings are the first sign that the types are regions of a continuum rather than islands, which Claim 4 measures (Ding et al. 2010).

## Claim 2 · Other algorithms find the same shapes; the frequency description matters more than the algorithm

**Where to click.** View `grid` → `algorithms`: position #1 under convex NMF, k-means and Ward at the same K. Then `feature sets`: position #1 on 15, 5, 2 bands and HFA. Switching a run in the matrix keeps the position and the run note prints the ARI against the run you came from.

![position 1 across algorithms](renders/04_pos1_cnmf.png)

![position 1 across algorithms](renders/04_pos1_kmeans.png)

![position 1 across algorithms](renders/04_pos1_hierarchical.png)

![position 1 across feature sets](renders/05_pos1_concat_bands15.png)

![position 1 across feature sets](renders/05_pos1_concat_bands5.png)

![position 1 across feature sets](renders/05_pos1_concat_bands2.png)

![position 1 across feature sets](renders/05_pos1_concat_hfa.png)

![ARI](panels/p0_ari.png)

**What you see.** The three algorithms put the production type on the same contacts; across feature sets the same position is a different set of contacts once the bands are coarse.

**The numbers.** ARI 0.51 (k-means vs convex NMF), 0.33, 0.26 across algorithms; 0.38 to 0.07 across feature sets; all three algorithms agree on 47 % of the electrodes, all four feature sets on 11 % (Hubert & Arabie 1985).

**What it means.** Changing the algorithm moves the electrodes less than changing the frequency description (Hubert & Arabie 1985). Agreement is not correctness: four descriptions can agree on an artefact of the gate.

## Claim 3 · The shapes come back when patients are resampled, and K 7 is the last K before one patient owns a cluster

**Where to click.** K `8`: one chip now carries the red * (a patient holds more than half of it); back to K `7` it is gone. Then View `held-out` → colour `consensus margin`, and source `patient halves` with the patient list at the bottom of the panel.

![K 8 one-patient cluster](renders/06_k8_one_patient.png)

![consensus margin](renders/07_consensus_margin.png)

![misses under the halves test](renders/09_misses_halves.png)

![patients](panels/p4_patients.png)

**What you see.** At K 8 the extra cluster is a single patient's strip; at K 7 none is. Under consensus margin the dark electrodes stay with their type when 20 % of the patients are dropped and the light ones switch; under the halves test the misses concentrate in a few patients.

**The numbers.** At K 8 one cluster is 100 % one patient; at K 7 the worst is 39 % (PAT_3965). PAC 0.234 against 0.428 ± 0.031 for the Gaussian cloud (Monti et al. 2003; Şenbabaoğlu et al. 2014). Per-type consensus c4 low-frequency rise 0.81, c5 production 0.78, c2 broadband decrease 0.78, c6 response-period low-frequency fall 0.71, c0 beta decrease 0.69, c1 audio-stimulus low-frequency 0.62, c3 sentence-stimulus low-frequency 0.53 (Monti et al. 2003). The halves placement keeps 62 % overall; EL051 2.0×, PAT_6619 1.5×, EL038 1.3× carry more than their share of the misses, 1 of 29 patients keep at least 80 % (Lange et al. 2004; Varoquaux et al. 2017).

**What it means.** The partition is stable against patients, and K 7 is where that stops; the cloud has no patient structure, so the dominance rule is the guard (Şenbabaoğlu et al. 2014).

## Claim 4 · The types are separable: an electrode from an unseen patient lands in its type nine times in ten, and the leaks run between neighbours

**Where to click.** View `held-out` (source `leave one patient out`, colour `where misses go`). Read the three numbers at the top of the panel and the per-type bars; then click the cell `c3 → c4` in matrix B (patient halves), and switch colour to `hit rate`.

![one number](panels/p1_one_number.png)

![per type](panels/p2_per_type.png)

![misses LOPO](renders/08_misses_lopo.png)

![confusions](panels/p3_confusions.png)

![cell c3 to c4](renders/11_cell_c3_to_c4_halves.png)

![hit rate](renders/10_hit_rate_lopo.png)

**What you see.** Most electrodes are small grey hits; the coloured misses sit between the territories of the two types they move between, and the clicked cell shows exactly which c3 electrodes the halves test places in c4.

**The numbers.** 90 % of held-out electrodes land in their published type with one patient out, 62 % with half the patients out, chance about 52 % (Tibshirani & Walther 2005; Lange et al. 2004). Recall per type c4 low-frequency rise 0.97, c0 beta decrease 0.94, c5 production 0.92, c2 broadband decrease 0.91, c6 response-period low-frequency fall 0.89, c1 audio-stimulus low-frequency 0.80, c3 sentence-stimulus low-frequency 0.78; c3 sentence-stimulus low-frequency loses most, to c4 (7 %) and under the halves test to c4 (23 %) (Tibshirani & Walther 2005; Kuhn 1955). Per electrode, 1354 of 1587 land in their type in every fold.

**What it means.** Separable in the sense that matters, an unseen electrode can be told apart, and only partly: the leaky types lose a fifth of their electrodes to their neighbours, which is what a continuum of responses looks like under a partition (Tibshirani & Walther 2005).

## Claim 5 · The scorecard: c2 and c4 solid, c1 and c3 fragile, c5 solid under patients but weak across methods

**Where to click.** In the held-out view click the type bar `c3` and then `c2` (or their chips): the brain keeps that type's electrodes only. Then the worst patient in the list at the bottom.

![c3 only](renders/12_row_c3_lopo.png)

![c2 only](renders/13_row_c2_lopo.png)

![worst patient](renders/14_patient_EL051_halves.png)

![scorecard](panels/p5_scorecard.png)

**What you see.** c3's electrodes are scattered and a fifth of them are coloured as misses; c2's are a tight temporal set with almost none.

**The numbers.** Across algorithms / feature sets (chance-adjusted overlap): c5 0.34 / 0.26, c0 0.52 / 0.24, c2 0.71 / 0.45, c6 0.49 / 0.02, c4 0.47 / 0.21, c1 0.18 / 0.18, c3 0.33 / 0.16 (Hubert & Arabie 1985; Kuhn 1955); consensus and held-out recall as above (Monti et al. 2003; Tibshirani & Walther 2005).

**What it means.** Each type is rated on all three questions; the colour of a scorecard cell is its rank in the column, not a composite score.

## Two things to know when reading the page

- The page opens at K 8 (its default); every claim above is read at K 7, the paper's cut. The held-out view switches to K 7 by itself.
- At K 7 the page's chips count the published fit's labels (c5 206, c2 164 …) while the paper's figures and the held-out view use the sweep fit's partition (c5 200, c2 168 …); the two agree on 95.7 % of the electrodes.

## References

- Tibshirani R, Walther G (2005). Cluster validation by prediction strength. Journal of Computational and Graphical Statistics 14(3): 511–528.
- Lange T, Roth V, Braun ML, Buhmann JM (2004). Stability-based validation of clustering solutions. Neural Computation 16(6): 1299–1323.
- Varoquaux G, Raamana PR, Engemann DA, Hoyos-Idrobo A, Schwartz Y, Thirion B (2017). Assessing and tuning brain decoders: cross-validation, caveats, and guidelines. NeuroImage 145: 166–179.
- Kuhn HW (1955). The Hungarian method for the assignment problem. Naval Research Logistics Quarterly 2: 83–97.
- Ding C, Li T, Jordan MI (2010). Convex and semi-nonnegative matrix factorizations. IEEE Transactions on Pattern Analysis and Machine Intelligence 32(1): 45–55.
- Hubert L, Arabie P (1985). Comparing partitions. Journal of Classification 2: 193–218.
- Monti S, Tamayo P, Mesirov J, Golub T (2003). Consensus clustering: a resampling-based method for class discovery and visualization of gene expression microarray data. Machine Learning 52: 91–118.
- Şenbabaoğlu Y, Michailidis G, Li JZ (2014). Critical limitations of consensus clustering in class discovery. Scientific Reports 4: 6207.
- Gretton A, Borgwardt KM, Rasch MJ, Schölkopf B, Smola A (2012). A kernel two-sample test. Journal of Machine Learning Research 13: 723–773.
- Rousseeuw PJ (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics 20: 53–65.
- Owen AB, Perry PO (2009). Bi-cross-validation of the SVD and the nonnegative matrix factorization. Annals of Applied Statistics 3(2): 564–594.
