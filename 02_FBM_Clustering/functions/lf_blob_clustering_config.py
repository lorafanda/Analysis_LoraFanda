
# blob_clustering_config.py

from typing import Dict, List

# -----------------------------
# Valley-blob segmentation
# -----------------------------

MAX_BLOBS: int = 6
THR_POS: float = 2.5
THR_NEG: float = -3.5

DELTA_VALLEY: float = 1.5
MIN_MEAN_POS_FACTOR: float = 0.7   # min_mean_pos = factor * THR_POS
MAX_MEAN_NEG_FACTOR: float = 0.8   # max_mean_neg = factor * THR_NEG

VALLEY_PARAMS: Dict[str, float | int | str] = {
    "thr_pos":      THR_POS,
    "thr_neg":      THR_NEG,
    "delta_valley": DELTA_VALLEY,
    "min_mean_pos": MIN_MEAN_POS_FACTOR * THR_POS,
    "max_mean_neg": MAX_MEAN_NEG_FACTOR * THR_NEG,
    "max_blobs":    MAX_BLOBS,
    "sign_mode":   "both",
}

# -----------------------------
# Score gating
# -----------------------------

# Your preferred hard minimum; can override quantile-based threshold
# SCORE_MIN_DEFAULT: float = 1000000.0

# Quantile used to *propose* a threshold from the score distribution
BLOB_SCORE_QUANTILE: float = 0.90

# -----------------------------
# Visualization defaults
# -----------------------------

VALLEY_OVERLAY_MAX_SAMPLES: int = 500
VALLEY_OVERLAY_N_COLS: int = 8

VALLEY_DEBUG_N_SAMPLES: int = 500
VALLEY_DEBUG_SAMPLE_STRIDE: int = 50
VALLEY_DEBUG_MAX_FRAMES: int = 120

# -----------------------------
# Clustering algorithms
# -----------------------------

RANDOM_STATE: int = 42

# K-ranges for KMeans and GMM
KMEANS_K_RANGE: List[int] = list(range(13, 26))  # 6–36 inclusive
GMM_K_RANGE:    List[int] = list(range(13, 26))

# HDBSCAN parameter grid (you can tune this)
HDBSCAN_PARAM_GRID = [
    {"min_cluster_size": 5, "min_samples": None},
    {"min_cluster_size": 10, "min_samples": None},
    {"min_cluster_size": 15, "min_samples": None},
    {"min_cluster_size": 20, "min_samples": None},
]

# Number of null replicates for silhouette null distributions
N_NULL_SIL: int = 200

# -----------------------------
# Blob-feature layout (8 dims / blob)
# -----------------------------
FEATURES_PER_BLOB = 8

FEATURE_NAMES: List[str] = [
    "t_start",
    "f_peak",
    "t_peak",
    "sf",
    "st",
    "cov",
    "mean",
    "area_norm",
]

# Per-feature weights (applied to each blob in X_blob)
FEATURE_TYPE_WEIGHTS = {
    "t_start":   50.0,
    "f_peak":    0.1,
    "t_peak":    2.5,
    "sf":        0.5,
    "st":        0.5,
    "cov":       0.2,
    "mean":      0.0,
    "area_norm": 0.005,
}


# # blob_clustering_config.py

# from typing import Dict, List

# # -----------------------------
# # Valley-blob segmentation
# # -----------------------------

# MAX_BLOBS: int = 10

# THR_POS: float = 3.0
# THR_NEG: float = -4.0

# DELTA_VALLEY: float = 1.5
# MIN_MEAN_POS_FACTOR: float = 0.7   # min_mean_pos = factor * THR_POS
# MAX_MEAN_NEG_FACTOR: float = 0.7   # max_mean_neg = factor * THR_NEG

# VALLEY_PARAMS: Dict[str, float | int | str] = {
#     "thr_pos":      THR_POS,
#     "thr_neg":      THR_NEG,
#     "delta_valley": DELTA_VALLEY,
#     "min_mean_pos": MIN_MEAN_POS_FACTOR * THR_POS,
#     "max_mean_neg": MAX_MEAN_NEG_FACTOR * THR_NEG,
#     "max_blobs":    MAX_BLOBS,
#     "sign_mode":   "both",
# }

# # -----------------------------
# # Score gating
# # -----------------------------

# # Hard cutoff you like to use for gating (can override in notebook)
# SCORE_MIN_DEFAULT: float = 500.0

# # Quantile used to *propose* a threshold from the score distribution
#  BLOB_SCORE_QUANTILE: float = 0.80

# # -----------------------------
# # Visualization defaults
# # -----------------------------

# VALLEY_OVERLAY_MAX_SAMPLES: int = 500
# VALLEY_OVERLAY_N_COLS: int = 8

# VALLEY_DEBUG_N_SAMPLES: int = 500
# VALLEY_DEBUG_SAMPLE_STRIDE: int = 50
# VALLEY_DEBUG_MAX_FRAMES: int = 120

# # -----------------------------
# # KMeans / clustering
# # -----------------------------

# RANDOM_STATE: int = 42
# KMEANS_K_RANGE: List[int] = list(range(6, 37))  # 6–36 inclusive

# # -----------------------------
# # Blob-feature layout (8 dims / blob)
# # -----------------------------

# FEATURE_NAMES: List[str] = [
#     "t_start",
#     "f_peak",
#     "t_peak",
#     "sf",
#     "st",
#     "cov",
#     "mean",
#     "area_norm",
# ]

# # Per-feature weights (applied to each blob in X_blob)
# FEATURE_TYPE_WEIGHTS = {
#     "t_start":   50.0,
#     "f_peak":    0.0,
#     "t_peak":    2.5,
#     "sf":        0.5,
#     "st":        0.5,
#     "cov":       0.2,
#     "mean":      0.0,
#     "area_norm": 0.005,
# }
