# overfactor_stability/

Everything written by **notebook 238** and by **`measure_cluster_stability.py`**.

One measure lives here: **Hennig (2007) set-overlap Jaccard under a bootstrap**.
Resample the electrodes with replacement, refit at the same K, and take the overlap
between each original cluster and its best match in the refit. Every row carries
`measure = hennig_bootstrap_jaccard_inbag` so it cannot be read as anything else.

Hennig's thresholds — **> 0.75 stable, 0.60–0.75 a pattern, < 0.60 not a real
cluster** — apply to the numbers in this folder and to nothing else in the project.

## Not to be confused with

| | what it measures | where |
|---|---|---|
| **here** | overlap between two *sets* of electrodes, across a bootstrap refit | `boot_*` in this folder |
| `per_cluster_stability.csv` | **Monti consensus** — how often *pairs* of electrodes land together, under subsampling **without** replacement | inside each run dir |
| `consensus_matrix.npy` | the full pairwise co-occurrence matrix behind the above | inside each run dir |
| `overfactor_filter_*` | this analysis **before** the 2026-08-21 correction | `../comparison/`, superseded |

The first two are both called "Jaccard stability" in this project and are **not on the
same scale**. One is an overlap between two sets; the other is a frequency of
co-occurrence between pairs. A value of 0.7 does not mean the same thing in each, and
Hennig's cut-offs do not transfer to the Monti one. `functions/lf_stability.py`
computes the Monti measure and its docstring cites Hennig — that citation is loose;
the procedure it implements is Monti et al. consensus clustering.

## The 2026-08-21 correction

Everything in `../comparison/overfactor_filter_*` was produced by a bootstrap whose
maximum attainable value was **1 − 1/e = 0.632**, because rows the resample never drew
were left in the Jaccard denominator while being unable to enter the numerator. Those
results were compared against 0.60 and 0.75 and were therefore near-guaranteed to fail.
Feeding that procedure a *perfect* reproduction of the partition returned 0.6322 ±
0.0084 instead of 1.0.

They are kept rather than deleted, are never read by any cell, and are listed as
superseded by section 8 of notebook 238. **Nothing in `../comparison/` should be
quoted.**

## Files

    boot_ALL.csv                       long form: one row per component per run
    boot_<method>_<featureset>_K<k>.csv    one run
    boot_<method>_<featureset>_K<k>.png     the same run as a figure
    boot_cluster_stability_K<k>.json   measure_cluster_stability.py, all four measures
    split_half_reliability.csv         a per-ELECTRODE measure, not a cluster one

### Columns in the CSVs

    boot_J, boot_lo, boot_hi   mean Jaccard and the 95% CI OF THE MEAN
    pass_bootstrap             boot_lo >= 0.60 -- read the LOWER BOUND, not the mean
    null_p95, above_null       what structureless data with the same covariance
                               reaches at this K; below it the cluster is at chance
    init_J, pass_init          reproducibility across random starts (Ward: blank,
                               it is deterministic and would score 1.0 by construction)
    patient_max, pass_patient  largest single-patient share (weight for cNMF,
                               members for a hard partition -- different quantities)
    interpret                  passed all of the above
    pct_added                  share of the cluster that only exists because the
                               responsiveness gate was lifted (ungated runs only)

A cluster whose interval straddles 0.60 has **not failed** — it is unresolved, and
needs more resamples before a verdict.
