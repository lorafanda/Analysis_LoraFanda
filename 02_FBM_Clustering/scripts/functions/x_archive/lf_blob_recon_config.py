# lf_blob_recon_config.py
from pathlib import Path

# ---------------------------------------------------------------------
# Core paths
# ---------------------------------------------------------------------

# Root with PAT_XXXX bundles and fsaverage directory
SHARE_PAT_ROOT = Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\#SHARE\To_send_collaborators")

# FreeSurfer-style fsaverage subject
FSAVERAGE_DIR = SHARE_PAT_ROOT / "fsaverage"

# Where to write coverage figures
COVERAGE_OUTPUT_ROOT = Path(
    r"\\nasac-m2.unige.ch\m-HumanNeuronLab\ANALYSIS\FLM\Analysis_Lora"
    r"\01_FBM_Analysis\outputs\Paper1_recons"
)

# ---------------------------------------------------------------------
# BIDS electrode files
# ---------------------------------------------------------------------

# Subdir inside PAT_XXXX containing BIDS
BIDS_ELECTRODES_SUBDIR = "BIDS"

# File name pattern for electrodes table in BIDS
# -> PAT_3415/BIDS/ieeg/sub-3415_electrodes.tsv  (primary)
# or PAT_3415/BIDS/sub-3415_electrodes.tsv       (fallback)
BIDS_ELECTRODES_TEMPLATE = "sub-{num}_electrodes.tsv"

# Coordinate space label (as stored in the BIDS file)
BIDS_COORD_SPACE = "mri"  # keep generic; not used for maths

# Optional: manual overrides for specific patients
# Example:
# CONTACTS_PATH_OVERRIDES = {
#     "PAT_3415": r"X:\path\to\custom_P3415_contacts.tsv",
# }
CONTACTS_PATH_OVERRIDES: dict[str, str] = {}

# ---------------------------------------------------------------------
# Patients to include by default
# ---------------------------------------------------------------------

PATIENT_IDS_DEFAULT = [
    "PAT_2868",
    "PAT_3066",
    "PAT_3301",
    "PAT_3390",
    "PAT_3455",
    "PAT_3415",
    "PAT_3975",
    "PAT_3965",
]

# ---------------------------------------------------------------------
# PyVista visualization style
# ---------------------------------------------------------------------

BRAIN_COLOR = "#ead6db"
BRAIN_OPACITY = 0.35
BRAIN_SPECULAR = 0.02
BRAIN_SPECULAR_POWER = 8
BRAIN_AMBIENT = 0.34
BRAIN_DIFFUSE = 0.66

SUBDURAL_RADIUS = 1.6
DEPTH_RADIUS = 1.2
ELECTRODE_OPACITY = 1.0

PATIENT_COLOR_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#1f78b4", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    "#17becf", "#aec7e8", "#98df8a", "#ffbb78", "#ffbb33",
]

# Which views to save for fsaverage coverage
VIEWS_TO_SAVE = ["left", "right", "frontal", "dorsal"]
VIEWS_TO_SAVE = ["xy", "xz", "yz", "yx", "zx", "zy", "iso"]
# ---------------------------------------------------------------------
# MNE / STC (optional)
# ---------------------------------------------------------------------

MNE_FSAV_SUBJECT = "fsaverage"
# subjects_dir must be the folder that CONTAINS the 'fsaverage' subdir
MNE_SUBJECTS_DIR = FSAVERAGE_DIR.parent

MNE_DEFAULT_SENSOR_COLOR = (1.0, 0.5, 0.0, 1.0)  # RGBA
MNE_3D_BACKEND = "pyvistaqt"

COVERAGE_SURF_NAME = "pial"
COVERAGE_MAX_DIST_MM = 5.0


# ---------------------------------------------------------------------
# fsaverage parcellation options
# ---------------------------------------------------------------------

# Which fsaverage parcellation to use by default for parcel-highlighted plots.
# Valid options (given your uploaded files) include:
#   "aparc"          -> lh.aparc.annot / rh.aparc.annot
#   "aparc.a2009s"   -> lh.aparc.a2009s.annot / rh.aparc.a2009s.annot
#   "aparc.DKTatlas" -> lh.aparc.DKTatlas.annot / rh.aparc.DKTatlas.annot
FSAVERAGE_PARCELLATION_DEFAULT = "aparc.a2009s"

# Root where fsaverage label/annot files live.
# Typically: <FSAVERAGE_DIR>/label
FSAVERAGE_LABEL_DIR = FSAVERAGE_DIR / "label"

# ---------------------------------------------------------------------
# ROI groups for parcel highlighting (Kilian etc.)
# ---------------------------------------------------------------------
# You fill this from your ROI definition (Kilian’s grouping).
# Structure:
#   FSAVERAGE_ROI_GROUPS = {
#       "GroupName1": {
#           "labels": [
#               "G_temp_sup-Lateral",    # example Destrieux names
#               "G_temp_sup-Plan_polar",
#               ...
#           ],
#           "color": "#e41a1c",
#       },
#       "GroupName2": {
#           "labels": [...],
#           "color": "#377eb8",
#       },
#       ...
#   }
#
# Label strings must match the names in the .annot file you choose
# (e.g. aparc.a2009s vs aparc vs DKT).
FSAVERAGE_ROI_GROUPS: dict[str, dict] = {
    # EXAMPLE ONLY – replace with your own groups
    # "Temporal_language": {
    #     "labels": [
    #         "G_temp_sup-Lateral",
    #         "G_temp_sup-Plan_polar",
    #         "G_temp_sup-Plan_tempo",
    #     ],
    #     "color": "#e41a1c",
    # },
}
