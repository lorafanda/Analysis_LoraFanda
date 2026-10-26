# Consensus clustering stability  -  concat_source_v17

written 2026-10-26 13:24 by consensus_stability.py; patients resampled (80 % without replacement), H = 200, null = covariance-matched Gaussian surrogate, 25 draws x 50 iterations, Ks 2..12.

## Chosen K per cell

| feature set | method | current K | chosen K (constrained) | unconstrained | constraint binding | PAC real | PAC null | RCSI | p | mean cluster consensus | clusters m(k) < 0.6 | ARI vs published | ARI vs refit (seed 42) | seed-rerun ARI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 bands | k-means | 7 | 6 | 6 | no | 0.176 | 0.287 | 0.487 | 0.038 | 0.811 | 0 | 0.942 | 0.942 | 0.955 |
| 15 bands | Ward | 7 | 4 | 12 | yes | 0.483 | 0.646 | 0.291 | 0.038 | 0.760 | 0 | - | 0.637 | - |
| 15 bands | convex NMF | 7 | 4 | 4 | no | 0.176 | 0.355 | 0.700 | 0.231 | 0.889 | 0 | - | 0.939 | 0.947 |
| 5 bands | k-means | 8 | 3 | 3 | no | 0.131 | 0.252 | 0.651 | 0.192 | 0.934 | 0 | - | 0.972 | 0.970 |
| 5 bands | Ward | 8 | 3 | 3 | no | 0.491 | 0.745 | 0.417 | 0.038 | 0.857 | 0 | - | 0.916 | - |
| 5 bands | convex NMF | 8 | 3 | 3 | no | 0.168 | 0.253 | 0.409 | 0.346 | 0.917 | 0 | - | 0.940 | 0.946 |
| 2 bands | k-means | 10 | 4 | 4 | no | 0.173 | 0.217 | 0.229 | 0.308 | 0.903 | 0 | - | 0.963 | 0.959 |
| 2 bands | Ward | 10 | 8 | 9 | yes | 0.336 | 0.402 | 0.180 | 0.038 | 0.593 | 5 | 0.668 | 0.668 | - |
| 2 bands | convex NMF | 10 | 6 | 6 | no | 0.259 | 0.419 | 0.482 | 0.077 | 0.728 | 0 | 0.844 | 0.884 | 0.937 |
| HFA | k-means | 6 | 3 | 3 | no | 0.127 | 0.227 | 0.582 | 0.269 | 0.940 | 0 | - | 0.959 | 0.940 |
| HFA | Ward | 6 | 2 | 2 | no | 0.222 | 0.654 | 1.081 | 0.038 | 0.916 | 0 | - | 0.647 | - |
| HFA | convex NMF | 6 | 3 | 12 | yes | 0.232 | 0.243 | 0.046 | 0.423 | 0.900 | 0 | - | 0.945 | 0.898 |

## Cross-method agreement of the consensus partitions at the current K (ARI), beside the single-fit values of the statistics script

| feature set | K | pair | ARI consensus | ARI single fit |
|---|---|---|---|---|
| 15 bands | 7 | k-means vs Ward | 0.409 | 0.332 |
| 15 bands | 7 | k-means vs convex NMF | 0.489 | 0.509 |
| 15 bands | 7 | Ward vs convex NMF | 0.271 | 0.258 |
| 5 bands | 8 | k-means vs Ward | 0.396 | 0.339 |
| 5 bands | 8 | k-means vs convex NMF | 0.355 | 0.297 |
| 5 bands | 8 | Ward vs convex NMF | 0.326 | 0.253 |
| 2 bands | 10 | k-means vs Ward | 0.377 | 0.365 |
| 2 bands | 10 | k-means vs convex NMF | 0.437 | 0.313 |
| 2 bands | 10 | Ward vs convex NMF | 0.371 | 0.284 |
| HFA | 6 | k-means vs Ward | 0.605 | 0.392 |
| HFA | 6 | k-means vs convex NMF | 0.499 | 0.281 |
| HFA | 6 | Ward vs convex NMF | 0.441 | 0.242 |

## Feature-set order

The existing tests ordered the sets 15 bands > 5 bands > 2 bands > HFA on placement and agreement. Here (mean over the three methods at the chosen K):

| feature set | mean RCSI | mean cluster consensus |
|---|---|---|
| 15 bands | 0.493 | 0.820 |
| 5 bands | 0.492 | 0.902 |
| 2 bands | 0.297 | 0.741 |
| HFA | 0.570 | 0.918 |

RCSI order: HFA > 15 bands > 5 bands > 2 bands; cluster-consensus order: HFA > 5 bands > 15 bands > 2 bands.

## Per cell: cluster stability ranking and patients, at the chosen K

### 15 bands x k-means  (chosen K = 6, current K = 7)
clusters by consensus m(k): c0 n=321 m=0.87, c1 n=293 m=0.85, c4 n=300 m=0.84, c3 n=217 m=0.81, c2 n=214 m=0.76, c5 n=242 m=0.73
named patients (rank of mean own-cluster consensus, of all patients; value): PAT_3455 #24 (0.78), PAT_3965 #8 (0.85), PAT_3975 #29 (0.70), PAT_5515 #11 (0.84), PAT_5533 #14 (0.83), EL046 #17 (0.83)
patient dominance at the chosen K: worst cluster c2 has 30 % of its 214 electrodes from PAT_3965
seed rerun: ARI 0.955 -> ok
verification: symmetric in [0,1] True; min co-samples 101 (ok); X_train md5 identical across runs True

### 15 bands x Ward  (chosen K = 4, current K = 7)
clusters by consensus m(k): c0 n=252 m=0.86, c2 n=346 m=0.84, c1 n=468 m=0.71, c3 n=521 m=0.62
named patients (rank of mean own-cluster consensus, of all patients; value): PAT_3455 #1 (0.91), PAT_3965 #6 (0.85), PAT_3975 #10 (0.79), PAT_5515 #13 (0.75), PAT_5533 #20 (0.68), EL046 #25 (0.65)
patient dominance at the chosen K: worst cluster c0 has 38 % of its 252 electrodes from PAT_3975
verification: symmetric in [0,1] True; min co-samples 101 (ok); X_train md5 identical across runs True

### 15 bands x convex NMF  (chosen K = 4, current K = 7)
clusters by consensus m(k): c0 n=177 m=0.90, c2 n=474 m=0.90, c3 n=311 m=0.89, c1 n=625 m=0.87
named patients (rank of mean own-cluster consensus, of all patients; value): PAT_3455 #29 (0.81), PAT_3965 #1 (0.94), PAT_3975 #17 (0.88), PAT_5515 #5 (0.91), PAT_5533 #14 (0.89), EL046 #25 (0.86)
patient dominance at the chosen K: worst cluster c0 has 36 % of its 177 electrodes from PAT_3965
seed rerun: ARI 0.947 -> ok
verification: symmetric in [0,1] True; min co-samples 101 (ok); X_train md5 identical across runs True

### 5 bands x k-means  (chosen K = 3, current K = 8)
clusters by consensus m(k): c0 n=282 m=0.95, c1 n=671 m=0.94, c2 n=634 m=0.91
named patients (rank of mean own-cluster consensus, of all patients; value): PAT_3455 #15 (0.93), PAT_3965 #1 (0.97), PAT_3975 #11 (0.94), PAT_5515 #10 (0.94), PAT_5533 #27 (0.90), EL046 #17 (0.93)
patient dominance at the chosen K: worst cluster c0 has 27 % of its 282 electrodes from PAT_3975
seed rerun: ARI 0.970 -> ok
verification: symmetric in [0,1] True; min co-samples 101 (ok); X_train md5 identical across runs True

### 5 bands x Ward  (chosen K = 3, current K = 8)
clusters by consensus m(k): c0 n=280 m=0.94, c1 n=441 m=0.84, c2 n=866 m=0.79
named patients (rank of mean own-cluster consensus, of all patients; value): PAT_3455 #2 (0.94), PAT_3965 #1 (0.95), PAT_3975 #3 (0.94), PAT_5515 #9 (0.84), PAT_5533 #8 (0.84), EL046 #27 (0.76)
patient dominance at the chosen K: worst cluster c0 has 34 % of its 280 electrodes from PAT_3975
verification: symmetric in [0,1] True; min co-samples 101 (ok); X_train md5 identical across runs True

### 5 bands x convex NMF  (chosen K = 3, current K = 8)
clusters by consensus m(k): c1 n=677 m=0.93, c0 n=234 m=0.91, c2 n=676 m=0.91
named patients (rank of mean own-cluster consensus, of all patients; value): PAT_3455 #29 (0.87), PAT_3965 #9 (0.94), PAT_3975 #25 (0.89), PAT_5515 #11 (0.93), PAT_5533 #22 (0.90), EL046 #14 (0.93)
patient dominance at the chosen K: worst cluster c0 has 28 % of its 234 electrodes from PAT_3965
seed rerun: ARI 0.946 -> ok
verification: symmetric in [0,1] True; min co-samples 101 (ok); X_train md5 identical across runs True

### 2 bands x k-means  (chosen K = 4, current K = 10)
clusters by consensus m(k): c3 n=416 m=0.93, c0 n=244 m=0.92, c1 n=345 m=0.92, c2 n=582 m=0.85
named patients (rank of mean own-cluster consensus, of all patients; value): PAT_3455 #6 (0.92), PAT_3965 #12 (0.90), PAT_3975 #20 (0.88), PAT_5515 #3 (0.93), PAT_5533 #10 (0.92), EL046 #21 (0.87)
patient dominance at the chosen K: worst cluster c0 has 26 % of its 244 electrodes from PAT_3455
seed rerun: ARI 0.959 -> ok
verification: symmetric in [0,1] True; min co-samples 101 (ok); X_train md5 identical across runs True

### 2 bands x Ward  (chosen K = 8, current K = 10)
clusters by consensus m(k): c6 n=247 m=0.74, c5 n=291 m=0.62, c3 n=99 m=0.62, c1 n=166 m=0.59 UNSTABLE, c4 n=113 m=0.59 UNSTABLE, c7 n=274 m=0.56 UNSTABLE, c0 n=172 m=0.55 UNSTABLE, c2 n=225 m=0.46 UNSTABLE
named patients (rank of mean own-cluster consensus, of all patients; value): PAT_3455 #13 (0.59), PAT_3965 #22 (0.55), PAT_3975 #19 (0.56), PAT_5515 #23 (0.55), PAT_5533 #8 (0.62), EL046 #17 (0.56)
patient dominance at the chosen K: worst cluster c0 has 35 % of its 172 electrodes from PAT_3455
verification: symmetric in [0,1] True; min co-samples 101 (ok); X_train md5 identical across runs True

### 2 bands x convex NMF  (chosen K = 6, current K = 10)
clusters by consensus m(k): c0 n=200 m=0.80, c1 n=383 m=0.75, c2 n=368 m=0.74, c3 n=281 m=0.74, c5 n=212 m=0.68, c4 n=143 m=0.65
named patients (rank of mean own-cluster consensus, of all patients; value): PAT_3455 #2 (0.81), PAT_3965 #11 (0.75), PAT_3975 #17 (0.73), PAT_5515 #6 (0.78), PAT_5533 #4 (0.80), EL046 #28 (0.65)
patient dominance at the chosen K: worst cluster c0 has 30 % of its 200 electrodes from PAT_3455
seed rerun: ARI 0.937 -> ok
verification: symmetric in [0,1] True; min co-samples 101 (ok); X_train md5 identical across runs True

### HFA x k-means  (chosen K = 3, current K = 6)
clusters by consensus m(k): c0 n=300 m=0.96, c2 n=838 m=0.96, c1 n=449 m=0.90
named patients (rank of mean own-cluster consensus, of all patients; value): PAT_3455 #1 (0.98), PAT_3965 #6 (0.96), PAT_3975 #25 (0.92), PAT_5515 #13 (0.95), PAT_5533 #18 (0.94), EL046 #27 (0.90)
patient dominance at the chosen K: worst cluster c0 has 22 % of its 300 electrodes from PAT_3455
seed rerun: ARI 0.940 -> ok
verification: symmetric in [0,1] True; min co-samples 101 (ok); X_train md5 identical across runs True

### HFA x Ward  (chosen K = 2, current K = 6)
clusters by consensus m(k): c0 n=1343 m=0.95, c1 n=244 m=0.88
named patients (rank of mean own-cluster consensus, of all patients; value): PAT_3455 #22 (0.91), PAT_3965 #23 (0.91), PAT_3975 #13 (0.95), PAT_5515 #4 (0.97), PAT_5533 #12 (0.95), EL046 #24 (0.91)
patient dominance at the chosen K: worst cluster c1 has 28 % of its 244 electrodes from PAT_3455
verification: symmetric in [0,1] True; min co-samples 101 (ok); X_train md5 identical across runs True

### HFA x convex NMF  (chosen K = 3, current K = 6)
clusters by consensus m(k): c0 n=316 m=0.92, c2 n=690 m=0.90, c1 n=581 m=0.88
named patients (rank of mean own-cluster consensus, of all patients; value): PAT_3455 #2 (0.94), PAT_3965 #5 (0.93), PAT_3975 #22 (0.88), PAT_5515 #3 (0.94), PAT_5533 #24 (0.88), EL046 #10 (0.92)
patient dominance at the chosen K: worst cluster c0 has 21 % of its 316 electrodes from PAT_3455
seed rerun: ARI 0.898 -> raise H
verification: symmetric in [0,1] True; min co-samples 101 (ok); X_train md5 identical across runs True

## Runtime

total 12.0 h over 12 cells (sum of the cells' own clocks; a resumed run counts only what it computed).

## Existing outputs

186 existing files hashed before and after: unchanged.

## Reading

Cluster consensus below 0.6 is flagged unstable (a convention). RCSI compares the real PAC with the surrogate's; the surrogate has the data's covariance but no patient structure, so a K whose clusters follow patients can look stable against it - the dominance constraint and the per-patient tables are the guards. Numbers only; no interpretation beyond them.