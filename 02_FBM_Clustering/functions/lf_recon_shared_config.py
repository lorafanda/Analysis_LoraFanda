# lf_recon_shared_config.py
from pathlib import Path


# -------------------------
# Shared reconstruction root (fsaverage + PAT_XXXX live here)
# -------------------------
SHARED_RECON_ROOT = Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\#SHARE\To_send_collaborators")

# -------------------------
# Base + output roots
# -------------------------
# If running in Jupyter, code will use CWD as base unless you set this to an absolute path.
BASE_DIR_OVERRIDE = None

OUTPUTS_ROOT = Path(
    r"\\nasac-m2.unige.ch\m-HumanNeuronLab\ANALYSIS\FLM\Analysis_LoraFanda\02_FBM_Clustering\outputs"
)

RUN230_ROOT = OUTPUTS_ROOT / "230_blob_clustering_runs"      # contains <RUN_ID_230>/
RUN240_ROOT = OUTPUTS_ROOT / "240_blob_cluster_recon"        # will contain <RUN_ID_RECON>/

# lf_blob_recon_config.py (ADD)
RUN241_ROOT = OUTPUTS_ROOT / "241_atlas_activity_plots"

RUN241_FRAMES_DIR = "frames"
RUN241_VIDEOS_DIR = "videos"
RUN241_MANIFEST_NAME = "manifest.json"

# -------------------------
# 230 canonical inputs (per run)
# -------------------------
META_IN_KEEP_WITH_CLUSTERS = "df_keep_with_clusters.parquet"
X_IN_KEEP = "X_blob_keep.npy"  # optional (not needed for atlas renders)

# -------------------------
# 240 derived outputs (per recon run)
# -------------------------
INPUTS_SNAPSHOT_DIR = "inputs_snapshot"
ATLAS_CACHE_DIR = "atlas_cache"
RENDER_OUT_DIRNAME = "renders_AB"
REPORTS_DIR = "reports"

PDF_INCLUDE_CLEAN = True
PDF_INCLUDE_APARC = False

META_OUT_NAME   = "df_meta_contact_level.tsv"
COORDS_OUT_NAME = "electrode_coords_fsaverage_tkr.tsv"
QC_OUT_NAME     = "transform_qc_summary.tsv"

# -------------------------
# Recon artifact root (derived outputs)
# -------------------------
# Canonical location for recon artifacts (replaces legacy "Paper1_recons")
RECON250_ROOT = OUTPUTS_ROOT / "251_recon"

# Backwards-compatible name used throughout the codebase.
# DO NOT change downstream logic; only change this root.
PAPER1_RECONS_ROOT = RECON250_ROOT
BLOCK_NAME = "glassbrain"
CSV_PRODUCT = "coords"
CONTACTS_CSV_SUFFIX = "_contacts_tkrRAS.csv"  # expects {pid}{suffix}

# -------------------------
# FreeSurfer subject roots
# -------------------------
ROOT_PAT   = SHARED_RECON_ROOT
ROOT_MICRO = Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\MICROEPI")
ROOT_BERN_FS = Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_BERN\Reconstruction")

# Search root for EL Lookups (Lead-DBS-style)
BERN_EL_PROJECT_ROOT = ROOT_BERN_FS.parent  # \\...\\SEEG_EXPERIMENTS_BERN

# fsaverage (atlas)
FSAVERAGE_DIR = SHARED_RECON_ROOT / "fsaverage"

# -------------------------
# Meta collapsing to contact-level
# -------------------------
COLLAPSE_CLUSTER_STRATEGY = "mode"  # "mode" or "first"
FILTER_HIGH_ACTIVITY = False
HIGH_ACTIVITY_COL = "high_activity"

# Cluster column detection: any column starting with this prefix is considered a candidate
CLUSTER_COL_PREFIX = "cluster_"

# Condition column (meta)
CONDITION_COL_CANDIDATES = ["condition", "conditions"]

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
BRAIN_OPACITY_CLEAN = 0.25
BRAIN_OPACITY_APARC = 0.25
BRAIN_SPECULAR = 0.02
BRAIN_SPECULAR_POWER = 8
BRAIN_AMBIENT = 0.34
BRAIN_DIFFUSE = 0.66

DEPTH_RADIUS = 1.5
SUBDURAL_RADIUS = 1.2
ELECTRODE_OPACITY = 1

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
# Colors (CSS4 names) — as requested
# -------------------------
EL_COLOR_NAMES = ["blue","deepskyblue","cyan","teal","green","lime","powderblue","forestgreen","lightcyan","olivegreen"]
PAT_COLOR_NAMES = ["blueviolet","fuchsia","deeppink","crimson","pink","red","chocolate","gold","purple","saddlebrown","lemonchiffon","lavenderblush","lime","powderblue","forestgreen","lightcyan"]
MICRO_COLOR_NAMES = ["navy","darkslategray","black","darkred","darkolivegreen"]

CONDITION_COLOR_NAMES = ["red", "blueviolet", "deeppink"]

Clustering_dendrogram_colors = [
    "powderblue",
    "paleturquoise",
    "lightcyan",
    "lightblue",
    "lightsteelblue",
    "lavender",
    "thistle",
    "plum",
    "pink",
    "lightpink",
    "mistyrose",
    "peachpuff",
    "moccasin",
    "lemonchiffon",
    "lightyellow",
    "honeydew",
    "palegreen",
    "lightgreen",
    "aquamarine",
    "lightseagreen",
]

# -------------------------
# Cell 3: aparc labels + heatmaps + PDF report
# -------------------------
APARC_LABELS_CACHE_NAME = "electrode_aparc_labels_fsaverage.tsv"  # saved under atlas_cache/

# Heatmap limits (to keep plots readable)
HEATMAP_TOP_N_REGIONS = 40     # keep top N aparc labels globally, collapse rest -> "Other"
HEATMAP_TOP_N_PATIENTS = 25    # keep top N patients globally, collapse rest -> "Other"

# PDF
REPORT_PDF_NAME = "cluster_diagnostics_report.pdf"

# Which atlas views to embed per cluster page (must match your saved view names)
PDF_VIEWS = ["left", "right", "dorsal", "ventral"]

# Which render styles to embed
PDF_INCLUDE_CLEAN = True
PDF_INCLUDE_APARC = False