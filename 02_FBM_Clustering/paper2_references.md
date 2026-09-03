# Paper 2 — the methods we lean on, most relevant first

Companion to `paper2_references.bib` (same keys). For each: what it is, how this paper
uses it, and where it shows up. Written 2026-09-03 against the K=8 figures.

---

**1. Ding, Li & Jordan 2010 — convex NMF** · `dingLiJordan2010`
The clustering model. Convex NMF writes X ≈ G(WᵀX): every centroid is a convex
combination of real electrodes, so a type is always something an electrode actually
did, never an abstract direction. We renormalise G to sum to 1 per electrode, which
gives the graded loadings the brains are coloured by, and take the argmax for hard
labels. Behind every figure.

**2. Owen & Perry 2009 — bi-cross-validation** · `owenPerry2009`
Held-out variance with rows AND columns held out — the only cross-validation of a
factorisation whose curve can turn over, and therefore the only one that can nominate
a K. FIG 1A. We then show (FIG 1D) why its peak is not the K to report.

**3. Lipkin et al. 2022 — LanA** · `lipkin2022lana`
Probabilistic language atlas from precision fMRI in >800 people: for every location, the
probability that it is language-responsive in a typical brain. FIG 3 looks each
electrode up in it (1396 of 1693 covered) and asks whether any cluster sits higher
than a spatial null. It is a prior about a place, not a measurement in a patient — the
Introduction and Limitations should both say so.

**4. Hubert & Arabie 1985 — adjusted Rand index** · `hubertArabie1985`
Chance-corrected agreement between two partitions. The upper number in every cell of
FIG 2A/D, and the ARI-vs-K inset.

**5. Hennig 2007 — cluster-wise Jaccard stability** · `hennig2007`
Bootstrap a clustering, match each cluster to its nearest in the resample, report the
Jaccard per cluster; Hennig's 0.6 rule of thumb for "a real cluster" is the one the
site quotes. Our `sweep_stability --native` is this with the run's own method; it gives
the per-cluster Jaccard in FIG 2B/E and the self-agreement ceiling (cNMF on high gamma:
0.55; k-means 0.82).

**6. Monti et al. 2003 — consensus clustering** · `monti2003consensus`
Resample, recluster, count how often each pair of items lands together: the consensus
matrix. `sweep_stability` writes one per K (50 resamples of 80%); PAC is read from it.

**7. Șenbabaoğlu, Michailidis & Li 2014 — PAC** · `senbabaoglu2014pac`
The proportion of ambiguous clustering: the share of consensus values that are neither
near 0 nor near 1. Lower is better. Under each column of FIG 2B/E.
→ **Fix before submission**: the FIG 2 caption currently credits PAC to "SC3, Nature
Methods 2017". SC3 popularised the consensus approach in single-cell work; PAC itself is
this paper. Cite both, credit PAC here.

**8. Alexander-Bloch et al. 2018 — the spin test** · `alexanderBloch2018spin`
Why a map-to-map correlation needs a spatially constrained null: smooth brain maps have
far fewer independent samples than points, and a naive permutation inflates every
comparison. Their fix (rotate a spherical projection) is for surfaces; ours is the sEEG
analogue — roll labels along each electrode shaft — and this is the paper that
justifies needing one. FIG 3, both panels; the reason the c3 enrichment disappears.

**9. Vinh, Epps & Bailey 2010 — NMI** · `vinh2010nmi`
Normalised mutual information and its variants. The lower number in each FIG 2A/D
cell; reported beside ARI because ARI is dominated by the large clusters and NMI is not.

**10. Cutler & Breiman 1994 — archetypal analysis** · `cutlerBreiman1994`
Components on the convex hull of the data, memberships summing to 1. One of the four
algorithms in FIG 2's algorithm half (with cNMF, k-means, Ward), and its own explainer
on the site.

**11. Benjamini & Hochberg 1995 — FDR** · `benjaminiHochberg1995`
All FIG 3 q-values, across the 8 clusters, under both nulls.

**12. Kuhn 1955 — Hungarian assignment** · `kuhn1955hungarian`
Optimal one-to-one matching of clusters between two solutions (via
`scipy.optimize.linear_sum_assignment`). Behind FIG 1's block order and everything
in FIG 2. Its one-to-one nature is also why splits had to be marked separately.

**13. Ward 1963 — hierarchical clustering** · `ward1963`
**14. Lloyd 1982 — k-means** · `lloyd1982kmeans`
The comparison algorithms in FIG 2D–F. k-means was also the refit inside the original
(non-native) `sweep_stability`, which is why that number was wrong for cNMF runs.

**15. Kiselev et al. 2017 — SC3** · `kiselev2017sc3`
Consensus clustering as the field-standard robustness check in a Nature Methods
paper. Keep as the "this is what the field does" cite; see item 7 for the attribution.

**16. Burt et al. 2020 — BrainSMASH** · `burt2020brainsmash`
*Not used.* Variogram-matched surrogate maps — the other spatially constrained null.
Named in Methods to say why we chose a shaft-shift instead: sEEG contacts are sparse
points on lines, not a field, and a shift along the shaft preserves exactly the
structure that matters.

**17. Desikan et al. 2006 — Desikan–Killiany atlas** · `desikan2006`
Anatomical labels for electrodes in the earlier pipeline stage (region tables, the old
"anatomical purity" check). Not in the three paper figures; belongs in STAR Methods.

**18–20. Software** · `pedregosa2011sklearn`, `virtanen2020scipy`, `harris2020numpy`
scikit-learn (ARI, NMI, k-means, Ward), SciPy (assignment, Spearman), NumPy. For the
key resources table.

---

Not in this list: the clinical and language-mapping literature the Introduction's
`[REF]` slots need (ESM outcomes, high-gamma as a proxy for firing, passive-mapping
vs ESM agreement). That is a separate list, and one you know better than I do.
