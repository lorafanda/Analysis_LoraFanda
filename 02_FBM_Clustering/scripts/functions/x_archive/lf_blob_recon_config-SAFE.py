# lf_blob_recon_config.py
from pathlib import Path

# -------------------------
# Base + input locations
# -------------------------
# If running in Jupyter, code will use CWD as base unless you set this to an absolute path.
BASE_DIR_OVERRIDE = None  # e.g. r"\\nasac-m2.unige.ch\m-HumanNeuronLab\ANALYSIS\FLM\Analysis_Lora\02_FBM_Clustering\scripts"

# -------------------------
# Meta input (NEW location, algorithm-specific)
# -------------------------
CLUSTERING_ROOT = Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\ANALYSIS\FLM\Analysis_Lora\02_FBM_Clustering\outputs\clustering")
DEFAULT_ALGO_NAME = "kmeans"   # driver can override
META_IN_FILENAME = "df_meta_with_clusters.tsv"
DEFAULT_CLUSTER_COL_IN_META = "cluster_kmeans_blob_weighted_bestK"

# -------------------------
# Outputs
# -------------------------
ATLAS_INPUTS_ROOT_BASE = Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\ANALYSIS\FLM\Analysis_Lora\02_FBM_Clustering\outputs\atlas_inputs")
RENDER_OUT_DIRNAME = "atlas_renders_AB"

META_OUT_NAME   = "df_meta_contact_level.tsv"
COORDS_OUT_NAME = "electrode_coords_fsaverage_tkr.tsv"
QC_OUT_NAME     = "transform_qc_summary.tsv"

# -------------------------
# Contact sources (PAT/Micro)
# -------------------------
PAPER1_RECONS_ROOT = Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\ANALYSIS\FLM\Analysis_Lora\01_FBM_Analysis\outputs\Paper1_recons")
BLOCK_NAME = "glassbrain"
CSV_PRODUCT = "coords"
CONTACTS_CSV_SUFFIX = "_contacts_tkrRAS.csv"  # expects {pid}{suffix}

# -------------------------
# FreeSurfer subject roots
# -------------------------
ROOT_PAT   = Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_HUG")
ROOT_MICRO = Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\MICROEPI")
ROOT_BERN_FS = Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_BERN\FreesurferResults")

# Search root for EL Lookups (Lead-DBS-style)
BERN_EL_PROJECT_ROOT = ROOT_BERN_FS.parent  # \\...\\SEEG_EXPERIMENTS_BERN

# fsaverage (atlas)
FSAVERAGE_DIR = Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_HUG\fsaverage")

# -------------------------
# Meta collapsing to contact-level
# -------------------------
COLLAPSE_CLUSTER_STRATEGY = "mode"  # "mode" or "first"
FILTER_HIGH_ACTIVITY = False
HIGH_ACTIVITY_COL = "high_activity"

# -------------------------
# EL Lookup parsing
# -------------------------
LOOKUP_SHEET = "channels"
LOOKUP_TYPE_COL = "type"
LOOKUP_KEEP_TYPES = {"lead", "seeg", "depth"}  # when present
LOOKUP_LABEL_PREF = ["natus", "name"]
LOOKUP_NATIVE_COLS = ("native_x", "native_y", "native_z")
ALLOW_EL_MNI_IF_NATIVE_MISSING = False
LOOKUP_MNI_COLS = ("mni_x", "mni_y", "mni_z")

# Optional manual overrides: pid -> path to contacts file
CONTACTS_PATH_OVERRIDES = {
    # "EL037": r"\\path\\to\\EL037_contacts_tkrRAS.csv",
}

# -------------------------
# Rendering
# -------------------------
INCLUDE_DEPTH = True
INCLUDE_SUBDURAL = True
SUBDURAL_COL_CANDIDATES = ["isSubdural", "isSubdural ", "is_strip", "isGrid", "is_grid", "subdural"]

WINDOW_SIZE = (1200, 1000)
SS_SCALE = 2
TRANSPARENT_BG = True

BRAIN_COLOR = "#ead6db"
BRAIN_OPACITY_CLEAN = 0.35
BRAIN_OPACITY_APARC = 0.35
BRAIN_SPECULAR = 0.02
BRAIN_SPECULAR_POWER = 8
BRAIN_AMBIENT = 0.34
BRAIN_DIFFUSE = 0.66

DEPTH_RADIUS = 1.5
SUBDURAL_RADIUS = 1.2
ELECTRODE_OPACITY = 0.98

VIEWS_TO_SAVE = ["left", "right", "frontal", "posterior", "dorsal", "ventral"]

# Cluster renders
PLOT_ALL_CLUSTERS = True
PLOT_ONE_CLUSTER_EACH = True
PLOT_PATIENT_COLORED_SUBFOLDERS_PER_CLUSTER = True
PLOT_CONDITION_COLORED_SUBFOLDERS_PER_CLUSTER = True

# Patient coverage mosaics inside each cluster folder
WRITE_PATIENT_COVERAGE = True
PATIENT_COVERAGE_VIEWS = ["left", "frontal", "right", "dorsal", "ventral"]  # 5 views
UNUSED_GRAY_RGB = (0.22, 0.22, 0.22)
PATIENT_COVERAGE_CACHE_DIRNAME = "_patient_coverage_cache"

# -------------------------
# Colors (CSS4 names) — exactly as requested
# -------------------------
EL_COLOR_NAMES = ["blue","deepskyblue","cyan","teal","green","lime","powderblue","forestgreen","lightcyan"]
PAT_COLOR_NAMES = ["blueviolet","fuchsia","deeppink","crimson","pink","red","yellow","chocolate","gold","purple","saddlebrown","lemonchiffon","lavenderblush"]
MICRO_COLOR_NAMES = ["navy","darkslategray","black","darkred","darkolivegreen"]

CONDITION_COLOR_NAMES = ["red", "blueviolet", "deeppink"]

# Condition column (meta)
CONDITION_COL_CANDIDATES = ["condition", "conditions"]
