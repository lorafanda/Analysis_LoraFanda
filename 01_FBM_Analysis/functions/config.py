# functions/config.py — Central settings for the ERSP pipeline

# ---------------------------
# Patients & blocks
# ---------------------------
patient_ids = [2868, 3066, 3301, 3390, 3415, 3455, 3965, 3975, "B-01","G-01", "G-02", "G-03", "G-04", 3780, "G-05"]
patient_ids = [3301, "B-01","G-01", "G-02", "G-03", "G-04", 3780, "G-05"]
# patient_ids = [3975,3780,2868,3301,  3066, 3390, 3415, 3455 ]
# patient_ids = [3415, 3455, 3975, 3301]
# patient_ids = [3415]
# patient_ids = [2868, 3066, 3301, 3390, 3415, 3455, 3965, 3975, 3780]
patient_ids = ["EL030","EL034","EL035","EL036","EL037","EL038","EL040","EL042","EL043","EL044","EL045",2868, 3066, 3301, 3390, 3415, 3455, 3965, 3975, 3780]
patient_ids = ["EL034","EL035","EL036","EL037","EL038","EL040","EL042"]
patient_ids = ["G-06", "G-04", "G-05","G-01", "G-02", "G-03","EL030","EL034","EL035","EL036","EL037","EL038","EL040" ,"EL042","EL043","EL044","EL045",2868, 3066, 3301, 3390, 3415, 3455, 3965, 3975, 3780]

patient_ids = ["G-06", "G-04", "G-05", "G-01", "G-02","G-03", "EL030","EL034","EL035","EL036","EL037","EL038","EL039","EL040","EL042","EL043","EL044","EL045", 2868, 3066, 3301, 3390, 3415, 3455, 3965, 3975, 3780]
# patient_ids = ["G-06", "G-04", "G-05"]#, 3415, 3455, 3965, 3975, 3780]
patient_ids = ["EL036","EL038","EL039"]


block_name  = "LM"
conditions_expected = ("picture", "audio", "reading")

COND_ALIAS = {
    "picture_naming":      "pict",
    "auditory_naming_ger": "audi",
    "auditory_naming_eng": "audi",
    "auditory_naming_fre": "audi",
    "reading_completion":  "read",
    "picture":             "pict",
    "auditory":            "audi",
    "reading":             "read",
}

# ---------------------------
# Referencing and Bad channels
# ---------------------------
reref_type = "WM"   # "WM" or "None"
# reref_type = "None"   # "WM" or "None"

# ---- Bad channels curation ----

EL_GRID_PATIENTS = {"EL044"}
EL_GRID_KEEP_PREFIXES = {"EL044": ("Pa", "postP", "T")}
# Patients with BOTH grid (ECoG) and depth (SEEG) electrodes. For these:
#   * WM reref still runs (uses depth-electrode WM contacts as usual)
#   * Channels with empty / "Unknown" tissueLabel that match one of these
#     prefixes are PROTECTED from the "drop Unknown" step — grid contacts
#     are legitimately unparcellated because BIDS parcellation only covers
#     channels that pierce cortex.
# bad_channels_manual still applies independently for excluding truly bad
# contacts (e.g. PAT_3415's HLG1-17 already in that list).
MIXED_GRID_DEPTH_PATIENTS = {"PAT_3415"}
MIXED_GRID_KEEP_PREFIXES = {
    # PAT_3415: 64 grid contacts GA1-GH8 (8x8 grid), 18 temporal strips
    # (TA/TM/TP × 1-6), 12 occipital strips (OI/OS × 1-6). G/T/O cover
    # all 94 surface contacts without matching HLG/IPG depth contacts.
    "PAT_3415": ("G", "T", "O"),
}

# ---- Channel-name hemisphere-strip override (EL043 etc.) ----
# Patients listed here have raw EDF channel names of the form
#   <electrode>_<L|R><number>     e.g. "A_L1", "A_L5", "A_L18"
# but their fsaverage coord CSV (e.g. EL043_contacts_fsaverage.csv) uses the
# BARE convention
#   name="A1", "A5", "A18"   +   hemi="L" tracked in its own column
# 140 normally keeps the underscore form (so "A_L1" → ERSP filename
# "..._A_L1_TN.tif" → labels.electrode = "A_L1" → 252 normalizes to "AL1"),
# which never matches the coord side "A1". For patients in this set, 140's
# channel iteration strips the `_<L|R>(?=\d)` infix early so:
#   * the WM marker can match TSV "A5" to EDF "A_L5" → "A5"
#   * ERSP filenames become "EL043_..._ERSP_A1_TN.tif"
#   * labels.csv electrode = "A1"
#   * 252's contact-name join matches "A1" ↔ coord "A1"
#
# DO NOT add patients whose coord CSV already bakes the hemisphere into the
# name (EL037/EL038/EL040/EL045 → "AL1", "aHR1", "OFR1", "AR1") — for them
# the current behaviour is already correct.
STRIP_HEMI_PATIENTS = {"EL043"}

bad_channels_manual = {
    "EL040": [ f"PlaT_L{i}" for i in range(1, 4)],  # example; fill after visual/clinical review
    "PAT_3415": [ f"HLG{i}" for i in range(1, 18)],  # example; fill after visual/clinical review
    "PAT_3780": ["FAP9"],  # example; fill after visual/clinical review
    "PAT_3975": ["TPG1"],  # example; fill after visual/clinical review
    "PAT_3965": ["cmd11", "y1", "y2", "y3", "y4"],
    "EL030": ["EntG_R18"],  # example; fill after visual/clinical review
    "EL036": [f"pHG_R{i}" for i in range(1, 13)],
    "EL037": ["pH_R7","aH_R1","A_R10","A_R12","CinG_L12","CinG_L13","aI_L18"]
              + [f"pI_L{i}" for i in range(1, 17)]
              + [f"pI_R{i}" for i in range(1, 17)],
    "EL038": ["pH_R7","aH_L3","STO_R1","A_L7", "CinG_R2"],  # example; fill after visual/clinical review
    "EL044": ["T57"]
}

# Placeholder for PSD suggestions you accept (copy printed line from driver here)
bad_channels_auto = {
    # "PAT_3415": ["WM2", "IPG4"],  # example; you paste/curate
}

# Line-noise / PSD heuristics
line_z = 3.0              # z-score cutoff for line/harmonic ratio
total_power_z = 3.5       # z-score cutoff for total power
flat_var_thresh = 1e-10   # near-flatline variance threshold

# WM reference building
wm_ref_method = "mean"  # robust median (or "mean")
wm_min_contacts = 3       # require at least this many clean WM contacts


# ---------------------------
# ERSP windows & warping
# ---------------------------
# Baseline in seconds (relative to onset)
baseline_w       = (-0.6, -0.1)
# If None -> use baseline_w; otherwise compute baseline stats in this tighter window
baseline_calc_w  = (-0.5, -0.1)

# Time-normalized proportions: (baseline, stim, post). Must sum to 1.
proportions = (0.0, 0.50, 0.50)
# proportions = (0.1, 0.45, 0.45)

# Real-time window (used only if mode="RT")
time_window = (-1.0, 7.0)

# Mode: "TN" (time-normalized) or "RT" (real time)
mode = "TN"

# Smoothing (use box; keep gaussian sigmas at 0)
smooth_kind   = "box"                   # <-- add this
box_kernel    = (3, 3)                  # <-- add this (try 3x3; adjust to taste)
smooth_passes = 1                       # <-- add this

# ---------------------------
# ERSP STFT params
# ---------------------------
nperseg    = 128
nfft       = nperseg * 2
noverlap   = int(0.85 * nperseg)  # ~75% (friendly default for Hann)
n_time_bins = 300                 # for TN
fmax       = 500.0                # plot y-limit
vmin, vmax = -6.0, 6.0            # dB display range

# ---------------------------
# Thresholding / clustering (used by driver via getattr)
# ---------------------------
# Detect on Z; display in dB
z_thresh          = 2.5                 # |Z| cutoff (two-sided)
thresh_mode       = "two-sided"         # "two-sided" | "pos" | "neg"

# Bin-based cluster filters
min_time_bins = 3     # e.g., 3–5 to enforce temporal extent
min_freq_bins = 3     # e.g., 4–8 depending on your df

# # Mild smoothing before thresholding (in bin units)
# sigma_t_bins      = 0
# sigma_f_bins      = 0

# # Cluster size filters (real units)
# min_duration_s    = 0.10                # minimum duration per blob
# min_bandwidth_hz  = 8.0                 # minimum bandwidth per blob

# # Channel-level summary flag (what fraction of bins must be significant)
# cluster_signif_min_frac = 0.01

# # ---------------------------
# # Trial filtering (if your collector uses these)
# # ---------------------------
# p_low      = 0.0       # percentile low for POST duration filter
# p_high     = 90.0      # percentile high
# min_stim_s = 0.5

# ---------------------------
# Outputs
# ---------------------------
outputs_root = r"\\nasac-m2.unige.ch\m-HumanNeuronLab\ANALYSIS\FLM\Analysis_LoraFanda\01_FBM_Analysis\outputs"
# script_name = "03_ersp_LM_20250923_masked"  # optional; your driver sets this itself

notch_patients  = ["G-06", "G-04", "G-05", "G-01","G-02", "G-03",
                   "EL030","EL034","EL035","EL036","EL040","EL042","EL043","EL044","EL045",
                   "PAT_3455","PAT_2868", "PAT_3066", "PAT_3301", "PAT_3390", "PAT_3415", "PAT_3965", "PAT_3975", "PAT_3780"]   
                    # IDs or substrings to match
    
# notch_patients  = ["3415","EL034","EL035","EL036","EL040","EL042"]   # IDs or substrings to match
mains_base      = 50.0       # 50 or 60
notch_Q         = 50         # 40–60 typical (only used if a non-adaptive notch path is wired in)
notch_repeats   = 1          # 1 is usually enough

# Adaptive notch: only apply notch at a harmonic if its z-score over the local
# spectrum background is at or above this threshold. Higher = more conservative
# (fewer harmonics notched, less risk of removing real signal).
notch_peak_z_thresh = 3.0

# ---------------------------
# Trial filtering & HG plot defaults
# ---------------------------
min_stim_s   = 0.5
max_post_s   = 10.0
iqr_k        = 1.5
hg_band      = (70.0, 150.0)
hg_smooth_ms = 25
hg_vmin      = -6.5
hg_vmax      = 6.5

## PATIENT CONFIGS ##

PAT_PATIENTS = [] #[2868, 3066, 3301, 3390, 3415, 3455, 3965, 3975, 3780]

PAT_PRESETS = {
    2868: dict(trig="PHOTO",  flip=False, time_range=(140, 1500), invalid_trials=[0,1,2,54,55,56,107,108,109], trial_ids=["picture"]*54 + ["auditory"]*53 + ["reading"]*53, fake_trials=[], manual_trig=None),
    3066: dict(trig="Xe1",   flip=False, time_range=(260, 1900), invalid_trials=[0,1,2,54,55,56,104,105,106,112], trial_ids=["picture"]*54 + ["auditory"]*50 + ["reading"]*53, fake_trials=[155], manual_trig=None),
    3301: dict(trig="PHOTO",  flip=False, time_range=(0, 1900),   invalid_trials=[], trial_ids=["picture"]*54, fake_trials=[], manual_trig="PAT_3301__FLM_picture__triggerPD.tsv"),
    3390: dict(trig="PHOTO",  flip=False, time_range=(170, 1222), invalid_trials=[0,1,2,54,55,56,107,108,109], trial_ids=["picture"]*54 + ["auditory"]*53 + ["reading"]*53, fake_trials=[], manual_trig=None),
    3415: dict(trig="photo",  flip=False, time_range=(30, 1500),  invalid_trials=[0,1,2,54,55,56,107,108,109], trial_ids=["picture"]*54 + ["auditory"]*53 + ["reading"]*53, fake_trials=[], manual_trig=None),
    3455: dict(trig="photo",  flip=False, time_range=(250, 1836), invalid_trials=[0,1,2,54,55,56,107,108,109], trial_ids=["picture"]*54 + ["auditory"]*53 + ["reading"]*53, fake_trials=[], manual_trig=None),
    3965: dict(trig="x1",    flip=False, time_range=(40, 1295),  invalid_trials=[0,1,2,54,55,56,107,108,109], trial_ids=["picture"]*54 + ["auditory"]*53 + ["reading"]*53, fake_trials=[], manual_trig=None),
    3975: dict(trig="E2",    flip=False, time_range=(200, 1550), invalid_trials=[], trial_ids=["picture"]*50 + ["auditory"]*50 + ["reading"]*51, fake_trials=[], manual_trig="PAT_3975__FLM_all.tsv"),
    3780: dict(trig="X2",    flip=False, time_range=(0, -1),     invalid_trials=[], trial_ids=["picture"]*51 + ["auditory"]*50 + ["reading"]*50, fake_trials=[], manual_trig="PAT_3780_FLM_all.tsv"),
}

# EL_PATIENTS = ["EL030","EL034","EL035","EL036","EL037","EL038","EL039","EL040","EL042","EL043","EL044","EL045"]
EL_PATIENTS = ["EL034","EL036","EL039","EL040","EL041","EL043","EL044","EL045"]

EL_PRESETS = {
    "EL030": dict(trig="DC6", flip=True, time_range=(250, 1600), invalid_trials=[0,1,2,54,55,56,107,108,109], trial_ids=["picture"]*54 + ["auditory"]*53 + ["reading"]*53, fake_trials=[], manual_trig="sub-EL030_task-LanguageMapping_timestamp-20-2-2024(13h57m48s)_lang-GER_events.tsv"),
    "EL034": dict(trig="DC6", flip=True, time_range=(0, -1),     invalid_trials=[], trial_ids=["picture"]*54 + ["auditory"]*53 + ["reading"]*53, fake_trials=[53,54,55,56,57,58,158], manual_trig=None),
    "EL035": dict(trig="DC6", flip=True, time_range=(0, -1),     invalid_trials=[0,1,2,54,55,56,107,108,109], trial_ids=["picture"]*54 + ["auditory"]*53 + ["reading"]*53, fake_trials=[160], manual_trig=None),
    "EL036": dict(trig="DC6", flip=True, time_range=(0, -1),     invalid_trials=[0,1,2,54,55,56,107,108,109], trial_ids=["picture"]*54 + ["auditory"]*53 + ["reading"]*53, fake_trials=[160,161], manual_trig=None),
    "EL037": dict(trig="DC6", flip=True, time_range=(0, -1),     invalid_trials=[0,1,2,54,55,56], trial_ids=["picture"]*54 + ["auditory"]*53 + ["reading"]*53, fake_trials=[160], manual_trig=None),
    "EL038": dict(trig="DC6", flip=True, time_range=(130, -1),   invalid_trials=[51,52,53,104,105,106], trial_ids=["picture"]*51 + ["auditory"]*56 + ["reading"]*53, fake_trials=[77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93], manual_trig="EL038_20241112_HUG_20250207_onsets_offsets_FBM_all_LM_.tsv"),
    "EL039": dict(trig="TRIG", flip=True, time_range=(0, -1),invalid_trials=[], trial_ids=["picture"]*54 + ["auditory"]*53 + ["reading"]*53, fake_trials=[160], manual_trig=None),
    "EL040": dict(trig="DC6", flip=True, time_range=(100, 1900), invalid_trials=[0,1,2,54,55,56,107,108,109], trial_ids=["picture"]*54 + ["auditory"]*53 + ["reading"]*53, fake_trials=[160], manual_trig=None),
    "EL042": dict(trig="DC6", flip=True, time_range=(0, -1),     invalid_trials=[0,1,2,54,55,56,107,108,109], trial_ids=["picture"]*55 + ["auditory"]*53 + ["reading"]*53, fake_trials=[160], manual_trig=None),
    "EL043": dict(trig="DC6", flip=True, time_range=(260, 3710), invalid_trials=[32,33,34,35,36,65,91,92,93,107,109,111,112,127,129,136], trial_ids=["picture"]*54 + ["auditory"]*53 + ["reading"]*53, fake_trials=[], manual_trig=None),
    "EL043": dict(trig="DC6", flip=True, time_range=(31, 863), invalid_trials=[], trial_ids=["picture"]*54, fake_trials=[], manual_trig=None),
    "EL044": dict(trig="DC6", flip=False, time_range=(2478, 3290),invalid_trials=[0,1], trial_ids=["picture"]*2 + ["auditory"]*53 + ["reading"]*0, fake_trials=[], manual_trig=None),
    "EL045": dict(trig="DC6", flip=False, time_range=(176, 1468), invalid_trials=[], trial_ids=["picture"]*54 + ["auditory"]*53 + ["reading"]*53, fake_trials=[], manual_trig=None),
}

# MICROEPI_PATIENTS = ["G-01","G-02","G-03","G-04","G-05"]

# MICROEPI_PRESETS = {
#     "G-01": dict(trig="ainp3", flip=False, time_range=(100, 1300), invalid_trials=[0,1,2,54,55,56,107,108,109], trial_ids=["picture"]*54 + ["auditory"]*53 + ["reading"]*53, fake_trials=[], manual_trig=None),
#     "G-02": dict(trig="ainp3", flip=False, time_range=(100, 1300), invalid_trials=[], trial_ids=["picture"]*54 + ["auditory"]*53 + ["reading"]*53, fake_trials=[], manual_trig=None),
#     "G-03": dict(trig="PHOTO", flip=False, time_range=(100, 1300), invalid_trials=[], trial_ids=["picture"]*54 + ["auditory"]*53 + ["reading"]*53, fake_trials=[], manual_trig=None),
#     "G-04": dict(trig="X2",    flip=False, time_range=(0, -1),     invalid_trials=[], trial_ids=["picture"]*51 + ["auditory"]*50 + ["reading"]*50, fake_trials=[], manual_trig="PAT_3780_FLM_all.tsv"),
#     "G-05": dict(trig="ainp3", flip=False, time_range=(100, 1300), invalid_trials=[], trial_ids=["picture"]*51 + ["auditory"]*50 + ["reading"]*50, fake_trials=[], manual_trig=None),
# }


# ---------------------------------------------------------------------------
# MicroEPI .mat → macro-only pipeline (G-04 / G-05 / G-06)
# ---------------------------------------------------------------------------
# These patients are processed in 140 via the MicroEPI .mat loader from
# `lf_micromacro.load_microepi_macros_for_pipeline`. Only macro channels
# (`dataEcog`) are used; micros are ignored. Outputs are saved under the
# `pat_name` (PAT_<bids_num>) so they sit alongside the regular PAT cohort
# downstream — same procedure, same output sizes.
# ---------------------------------------------------------------------------
# MicroEPI .mat → macro-only pipeline (G-01 through G-06)
# These patients are processed in 140 via the MicroEPI .mat loader from
# `lf_micromacro.load_microepi_macros_for_pipeline`. Only macro channels
# (`dataEcog`) are used; micros are ignored.
# ---------------------------------------------------------------------------

MICROEPI_MAT_PATIENTS = ["G-01", "G-02","G-03","G-04", "G-05", "G-06"] #"G-01", "G-02", "G-03",
MICROEPI_MAT_PATIENTS = [ ] #"G-01", "G-02", "G-03",

_NASAC           = r"\\nasac-m2.unige.ch\m-HumanNeuronLab"
_BIDS_ELEC_MICROEPI = _NASAC + r"\DATARAW\BIDS_elec\MICROEPI"
_MICROEPI_RAW    = _NASAC + r"\DATARAW\MICROEPI"

MICROEPI_MAT_PRESETS = {
    "G-01": {
        "pat_name":       "PAT_5515",      # adjust BIDS num if different
        "data_dir":       _MICROEPI_RAW+ r"\MicroEPI-G-01\task_otherlabs\exp_Lora1FLM_all\prep",
        "mat_files":      ["f0001_export_Labs_phmicrodown.mat",
                           "f0002_export_Labs_phmicrodown.mat"],
        "tsv_file":       "sub-MicroEpiG01_task-LanguageMapping_timestamp-28-11-2024(15h52m8s)_lang-FRE_events.tsv",
        "electrodes_tsv": _BIDS_ELEC_MICROEPI + r"\sub-5515\ieeg\*_electrodes.tsv",
        "trig":           "photodiode",
        "flip":           True,
        "time_range":     (190, 1614),
        "invalid_trials": [0,1,2,54,55,56,107,108,109],
        "trial_ids":      ["picture"]*54 + ["auditory"]*53 + ["reading"]*53,
        "fake_trials":    [],
    },
    "G-02": {
        "pat_name":       "PAT_5533",
        "data_dir":       _MICROEPI_RAW + r"\MicroEPI-G-02\task_otherlabs\Lora_FLM\prep",
        "mat_files":      ["f0001_export_Labs_ph.mat",
                           "f0002_export_Labs_ph.mat"],
        "tsv_file":       "sub-Microepi02_task-LanguageMapping_timestamp-16-1-2025(13h56m0s)_lang-FRE_events.tsv",
        "electrodes_tsv": _BIDS_ELEC_MICROEPI + r"\sub-5533\ieeg\*_electrodes.tsv",
        "trig":           "photodiode",
        "flip":           True,
        "time_range":     (10, 1295),
        "invalid_trials": [0,1,2,54,55,56,107,108,109],
        "trial_ids":      ["picture"]*54 + ["auditory"]*53 + ["reading"]*53,
        "fake_trials":    [],
    },
    "G-03": {
        "pat_name":       "PAT_6619",
        "data_dir":       _MICROEPI_RAW + r"\MicroEPI-G-03\task_FBM\exp3_Lora1_LM_CLA_CLV\prep",
        "mat_files":      ["f0001_export_Labs_ph.mat",
                           "f0002_export_Labs_ph.mat",
                           "f0003_export_Labs_ph.mat"],
        "tsv_file":       "sub-microepig03_task-LanguageMapping_timestamp-7-3-2025(14h59m34s)_lang-FRE_events.tsv",
        "electrodes_tsv": _BIDS_ELEC_MICROEPI + r"\sub-6619\ieeg\*_electrodes.tsv",
        "trig":           "photodiode",
        "flip":           True,
        "time_range":     (1690, 290),
        "invalid_trials": [0,1,2,54,55,56,107,108,109],
        "trial_ids":      ["picture"]*54 + ["auditory"]*53 + ["reading"]*53,
        "fake_trials":    [],
    },
    "G-04": {
        "pat_name":       "PAT_6704",
        "data_dir":       _MICROEPI_RAW + r"\MicroEPI-G-04\task_FBM\data_LM\exp4_Lora\prep",
        "mat_files":      ["f0001_export_Labs_phmicrodown.mat",
                           "f0002_export_Labs_phmicrodown.mat"],
        "tsv_file":       "sub-MicroEpi-G-04_task-LanguageMapping_timestamp-7-4-2025(14h55m53s)_lang-FRE_events.tsv",
        "electrodes_tsv": _BIDS_ELEC_MICROEPI + r"\sub-6704\ieeg\*_electrodes.tsv",
        "trig":           "photodiode",
        "flip":           False,
        "time_range":     (540, 1840),
        "invalid_trials": [0,1,2,54,55,56,107,108,109],
        "trial_ids":      ["picture"]*51 + ["auditory"]*50 + ["reading"]*59,
        "fake_trials":    [107,108,109,110,111,112,113,114,115,116],
    },
    "G-05": {
        "pat_name":       "PAT_6684",
        "data_dir":       _MICROEPI_RAW + r"\MicroEPI-G-05\tasks\exp9_JonathanFLM_2025_06_18\prep",
        "mat_files":      ["f0001_export_Labs_phmicrodown.mat",
                           "f0002_export_Labs_phmicrodown.mat"],
        "tsv_file":       "sub-microepi-g-05_task-LanguageMapping_datetime-18-6-2025(15h39m19s)_language-FRE_events.tsv",
        "electrodes_tsv": _BIDS_ELEC_MICROEPI + r"\sub-6684\ieeg\*_electrodes.tsv",
        "trig":           "photodiode",
        "flip":           True,
        "time_range":     (100, 1700),
        "invalid_trials": [0,1,2,54,55,56,107,108,109],
        "trial_ids":      ["picture"]*51 + ["auditory"]*50 + ["reading"]*50,
        "fake_trials":    [],
    },
    "G-06": {
        "pat_name":       "PAT_6854",
        "data_dir":       _MICROEPI_RAW + r"\MicroEPI-G-06\tasks\exp1_lora_2026_01_29_withmicro\prep",
        "mat_files":      ["f0001_export_Labs_phmicrodown.mat",
                           "f0002_export_Labs_phmicrodown.mat"],
        "tsv_file":       "sub-6854_task-LanguageMapping_datetime-29-1-2026(17h33m57s)_language-FRE_events.tsv",
        "electrodes_tsv": _BIDS_ELEC_MICROEPI + r"\sub-6854\ieeg\*_electrodes.tsv",
        "trig":           "photodiode",
        "flip":           True,
        "threshold":      0.75,
        "time_range":     (700, 2080),
        "invalid_trials": [0,1,2,54,55,56,107,108,109],
        "trial_ids":      ["picture"]*54 + ["auditory"]*53 + ["reading"]*53,  # adjust if needed
        "fake_trials":    [],
    },
}