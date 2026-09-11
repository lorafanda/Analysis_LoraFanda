"""Motor Mapping (MM) configuration - the stage-07 twin of 01_FBM_Analysis/functions/config.py.

WHAT MM IS. Five blocks of a movement task, one movement per block, cued by a photodiode
flash. Mouth Movement is the block this analysis exists for and Hand and Foot are its
built-in negative controls: if the outer IMG/IMD contacts are picking up jaw and
temporalis rather than brain, mouth movement must light them up and foot movement must
not. Nothing else in the cohort gives that contrast.

HOW EACH PATIENT'S FILES WERE IDENTIFIED, because it is not obvious and it is worth
recording. The acquisition date/time fields in the Micromed .TRC headers are ZEROED
(anonymised), and every file's mtime is the copy date, so neither the filename nor the
timestamp says which recording holds which experiment. The Blackrock session-stamp folder
is no help either - it is when the RECORDING SESSION opened, which can be two weeks before
the experiment. What does identify them is the experimenter's own Micromed note, typed
during the recording and stored in the TRC's NOTE zone with a sample index:

    G-01  EEG_2321937.TRC   9.96 min  168 ch  "motormapping"                    @    1.28 s
    G-01  EEG_2322500.TRC  21.66 min  168 ch  "motor mapping lora"              @    1.19 s
    G-02  EEG_2425653.TRC  24.02 min  180 ch  "Test Motor Mapping Lora"         @    0.06 s
    G-03  EEG_2491760.TRC  18.93 min  158 ch  "expe lora motor mapping start"   @    3.34 s
                                              "expe lora motor mapping end"     @ 1117.72 s
    G-05  EEG_2675077.TRC  14.64 min  150 ch  "expe motor mapping jon"          @    4.03 s

G-04 and G-06 have no motor-mapping marker in 222 and 439 TRC files respectively, and no
motor-mapping folder; they are not in this table.

WHICH FILES THE ANALYSIS READS: THE RAW ONES. No .mat export is used. The macro signal
comes from the .TRC, the micro signal and the cues from the Blackrock .ns6. The exports
were checked and hold nothing the raw files do not, with one exception noted below.

READING A .TRC: THE CHANNEL LABELS NEED THE ORDER ZONE. LABCOD holds 640 slots of 128
bytes, more than any patient has channels, and the ORDER zone maps acquisition index ->
slot. Taking LABCOD slot i as channel i gives labels that are right for the first few
channels and quietly wrong after that - which is how MKR1+ came to be read as an EEG
contact. Always go through ORDER.

TWO CLOCKS, NEVER ONE. The Micromed macro recording and the Blackrock micro recording
start and stop independently. G-05's TRC is 878.5 s while its f0001 export is ~1169 s -
the same experiment, two different time origins. Every time field in this file therefore
says which recording it belongs to. A window read off one and applied to the other is
silently wrong, and 1167 s is simply past the end of an 878.5 s file.

THE PHOTODIODE IS ON THE BLACKROCK ONLY. This was searched for, not assumed. Every one of
the Micromed's channels was tested for the 3.2 s square that the Blackrock plainly has:
the X* auxiliary channels are floating inputs correlated at r ~ 1.0 and peaking at 50 Hz
mains, the four MKR lines carry an identical featureless 1 Hz clock, and no macro channel
carries the cue. So cue times come from the Blackrock ainp1 for every patient, and
time_range is in the BLACKROCK session clock.

    patient   channel      p2p    coupling
    G-01      ainp1         80    AC, sawtooth droop
    G-02      photodiode    95    AC, sawtooth droop
    G-03      ainp1        604    DC, textbook square
    G-05      ainp1        143    DC, plus a 6661 calibration flash

WHY time_range MATTERS MORE HERE THAN IN LM. The detector normalises the derivative by its
maximum over the analysed window, so the threshold is a FRACTION OF THE LARGEST EDGE IN
THAT WINDOW. G-05's calibration flash is 47x its cue pulses inside one file; with the flash
in the window every real cue falls far below any threshold and the detector finds 9 events
instead of ~144. The fix is deliberately NOT a change to the detector - it is this file:
time_range excludes the flash, and pd_norm_blocks splits a drifting session into pieces
normalised separately. Same mechanism as pd_blocks in the LM config, same reason.

CROSS-SYSTEM ALIGNMENT HAS NO SHARED TRIGGER. The Blackrock's ~5.78 s digital word is
synchronisation hardware and reaches only the Blackrock; the Micromed's 1 Hz clock reaches
only the Micromed. Nothing is common to both except the brain, so alignment is measured by
cross-correlating a micro channel against its macro twin on the SAME shaft - AGm1 against
AG1 - which is valid because the two sit a millimetre apart and record the same local
field through two independently clocked amplifiers. It is checked by agreement across
anatomically distinct shafts, which have no other reason to agree on a lag.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# roots
# ---------------------------------------------------------------------------
_MICROEPI_RAW = r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\MICROEPI"
_BIDS_ELEC_MICROEPI = r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\BIDS_elec\MICROEPI"

# ---------------------------------------------------------------------------
# the task
# ---------------------------------------------------------------------------
block_name = "MM"

# The five blocks, exactly as the behavioural tsv spells them. Verified against every
# patient's own log rather than assumed: G-03 runs 30 trials per block (150 total),
# G-05 runs 20 (100 total), G-01 was run twice with different lengths (see its entry).
conditions_expected = (
    "Hand Movement (Left)",
    "Hand Movement (Right)",
    "Foot Movement (Left)",
    "Foot Movement (Right)",
    "Mouth Movement",
)

# Short machine-safe names, used for output folders and figure filenames. The parenthesised
# originals do not survive a filesystem intact on every platform, and "Mouth Movement" as a
# folder name invites a space-handling bug the first time it is passed to a shell.
COND_ALIAS = {
    "Hand Movement (Left)":  "hand_left",
    "Hand Movement (Right)": "hand_right",
    "Foot Movement (Left)":  "foot_left",
    "Foot Movement (Right)": "foot_right",
    "Mouth Movement":        "mouth",
}
COND_SHORT = tuple(COND_ALIAS[c] for c in conditions_expected)

# BLOCK ORDER IS FIXED. Every behavioural log that exists - G-01, G-03, G-05 - lists the
# five blocks in exactly this order, and the task sends the same five in the same order to
# every patient. That is what makes G-02 analysable without its log: its cue train has five
# blocks, and their order is known. For that patient the labelling rests on this assumption
# rather than on a log, which is why it is written down here instead of being implicit.
BLOCK_ORDER = conditions_expected

# The contrast the analysis is for. Mouth against Foot is the clean one: both are movements
# with the same cue and the same block structure, and only one of them involves the muscles
# that could contaminate a temporal or orbital contact.
CONTRAST_PRIMARY = ("mouth", "foot_left", "foot_right")
CONTRAST_CONTROL = ("hand_left", "hand_right")

# Design, from the behavioural logs: baseline ~3.1-3.3 s then a 3.0 s stimulus.
design_baseline_s = 3.2
design_stim_s = 3.0

# A DETECTED TRIAL WHOSE STIMULUS IS NOT ~3 s IS A DETECTION ERROR, NOT A SHORT TRIAL. The
# stimulus is a fixed 3.0 s by design and measures 3.02-3.09 s in practice, so anything
# outside this window is a missed or spurious edge - two cues merged, or an artefact
# counted as one. Such a trial has the wrong onset, and an ERSP built on a wrong onset is
# worse than one trial fewer.
trial_stim_min_s = 2.5
trial_stim_max_s = 3.5

# ---------------------------------------------------------------------------
# per-patient presets
# ---------------------------------------------------------------------------
# Keys mirror MICROEPI_MAT_PRESETS in the LM config so the same reader can serve both,
# plus the MM-only fields at the end of each entry.
#
#   pat_name        BIDS/PAT id, the name outputs are written under
#   data_dir        folder holding the Matlab exports
#   mat_files       the exports, in order
#   tsv_file        behavioural log, relative to beh_dir
#   beh_dir         folder holding tsv_file (LM keeps it in data_dir; MM often does not)
#   electrodes_tsv  BIDS electrodes table, for anatomy
#   trig            trigger channel name inside the export
#   flip            photodiode polarity, applied to the trace before detection
#   time_range      (t0, t1) SECONDS IN THE EXPORT'S CLOCK - the window the photodiode was
#                   plugged in and cueing. Excludes calibration flashes. None = whole file.
#   pd_norm_blocks  None, "auto", or [[t0,t1], ...] in the same clock. Each window is
#                   normalised separately before detection, for a photodiode that drifts.
#   trc             the Micromed file this session came from, for provenance
#   trc_marker      (start, end) marker text, or (start, None) where only a start was typed
#   trc_block_s     (t0, t1) seconds IN THE TRC'S CLOCK from those markers. Provenance only:
#                   do NOT apply these to the export, they are a different clock.
#   blackrock_dir   folder holding the .ns6/.nev parts
#   blackrock       (first, last) part numbers
#   n_trials_expected  from the behavioural log, so a mismatch is caught not absorbed
#   exclude_trials  indices to drop, 0-based, after the trial table exists
#   status          "ready" | "blocked" - and if blocked, why, in words
#
MM_PRESETS = {
    # -------------------------------------------------------------------
    "G-01": {
        "pat_name":       "PAT_5515",
        # TWO SESSIONS, ON PURPOSE. The first run got through Hand Left and Hand Right
        # only; it was re-run the next day to finish. Both are real data and both are
        # listed, which is why every other field here is a pair.
        "data_dir":       os.path.join(_MICROEPI_RAW, r"MicroEPI-G-01\task_motormapping\sync_micromedBlackrock"),
        "mat_files":      None,     # see status: this patient has no *_export_Labs_ph.mat
        "beh_dir":        os.path.join(_MICROEPI_RAW, r"MicroEPI-G-01\task_motormapping"),
        "tsv_file":       ["sub-01motormapping_task-motormapping_datetime-20241202-230732.tsv",
                           "sub-G01motormapping_task-motormapping_datetime-20241203-154039.tsv"],
        "electrodes_tsv": os.path.join(_BIDS_ELEC_MICROEPI, r"sub-5515\ieeg\*_electrodes.tsv"),
        "trig":           "photodiode",
        "flip":           True,
        "time_range":     None,     # TODO: photodiode plugged window, per session
        "pd_norm_blocks": None,
        "bk_pd_channel":  "ainp1",
        "bk_pd_p2p":      80,       # measured, AC-coupled with a sawtooth droop
        "trc":            ["EEG_2321937.TRC", "EEG_2322500.TRC"],
        "trc_marker":     [("motormapping", None), ("motor mapping lora", None)],
        "trc_block_s":    [(1.28, None), (1.19, None)],
        "blackrock_dir":  os.path.join(_MICROEPI_RAW, r"MicroEPI-G-01\task_motormapping\raw_blackrock"),
        "blackrock":      ("20241202-160318-253", "20241202-160318-259"),
        "n_trials_expected": [60, 100],
        "exclude_trials": [],
        "out_dir":        os.path.join(_MICROEPI_RAW, r"MicroEPI-G-01\task_FBM\data_MM"),
        "status":         "blocked: no f000N_export_Labs_*.mat for MM - this patient has "
                          "the older sync_micromedBlackrock export (f0003..f0007 split "
                          "into _ecog.mat and _micro.mat). Also two behavioural logs share "
                          "the 20241203 filename with DIFFERENT contents: the copy in "
                          "task_FBM/data_MM/raw has 85 trials (Mouth 5), the copy in "
                          "task_motormapping has 100 (Mouth 20). One of them is truncated "
                          "and it is not safe to guess which is the record.",
    },
    # -------------------------------------------------------------------
    "G-02": {
        "pat_name":       "PAT_5533",
        "data_dir":       os.path.join(_MICROEPI_RAW, r"MicroEPI-G-02\task_FBM\data_MM\sync_micromedBlackrock_tmp"),
        "mat_files":      None,     # see status
        "beh_dir":        None,     # see status
        "tsv_file":       None,
        "electrodes_tsv": os.path.join(_BIDS_ELEC_MICROEPI, r"sub-5533\ieeg\*_electrodes.tsv"),
        "trig":           "photodiode",
        "flip":           True,
        "time_range":     None,
        "pd_norm_blocks": None,
        "bk_pd_channel":  "photodiode",   # this rig labelled it; the others say ainp1
        "bk_pd_p2p":      95,              # measured, AC-coupled with a sawtooth droop
        "trc":            "EEG_2425653.TRC",
        "trc_marker":     ("Test Motor Mapping Lora", None),
        "trc_block_s":    (0.06, None),
        "blackrock_dir":  os.path.join(_MICROEPI_RAW, r"MicroEPI-G-02\task_FBM\data_MM\raw_blackrock"),
        "blackrock":      ("20250120-181120-713", "20250120-181120-717"),
        "n_trials_expected": None,
        "exclude_trials": [],
        "out_dir":        os.path.join(_MICROEPI_RAW, r"MicroEPI-G-02\task_FBM\data_MM"),
        "labels_from":    "block_order",   # NOT from a log - see BLOCK_ORDER above
        "status":         "ready (labels by block order, not by log)",
        "notes":          "No behavioural tsv exists for MM anywhere under this patient. "
                          "The five blocks are labelled by POSITION using BLOCK_ORDER, "
                          "which every other patient's log confirms. Two consequences to "
                          "keep in mind: the trial COUNT per block is unknown, so it is "
                          "taken from the cue train rather than checked against a log; and "
                          "if this session ever ran the blocks in a different order the "
                          "labels would be confidently wrong with nothing to catch it. "
                          "The Micromed marker reads 'Test Motor Mapping Lora', but 24 min "
                          "is a full-length run and the cue train is clean.",
    },
    # -------------------------------------------------------------------
    "G-03": {
        "pat_name":       "PAT_6619",
        "data_dir":       os.path.join(_MICROEPI_RAW, r"MicroEPI-G-03\task_FBM\exp11_Lora2_MM"),
        "mat_files":      ["f0001_export_Labs_ph.mat",
                           "f0002_export_Labs_ph.mat"],
        "beh_dir":        os.path.join(_MICROEPI_RAW, r"MicroEPI-G-03\task_FBM\exp11_Lora2_MM"),
        "tsv_file":       "sub-microepig03_task-motormapping_datetime-20250313-150809.tsv",
        "electrodes_tsv": os.path.join(_BIDS_ELEC_MICROEPI, r"sub-6619\ieeg\*_electrodes.tsv"),
        "trig":           "photodiode",
        "flip":           True,
        "time_range":     None,     # TODO: photodiode plugged window, in the EXPORT's clock
        "pd_norm_blocks": None,
        "bk_pd_channel":  "ainp1",
        "bk_pd_p2p":      604,      # measured, DC-coupled, a textbook square
        "trc":            "EEG_2491760.TRC",
        "trc_marker":     ("expe lora motor mapping start", "expe lora motor mapping end"),
        "trc_block_s":    (3.34, 1117.72),
        "blackrock_dir":  os.path.join(_MICROEPI_RAW, r"MicroEPI-G-03\raw_blackrock\20250307-201025"),
        "blackrock":      ("20250307-201025-1592", "20250307-201025-1596"),
        "n_trials_expected": 150,   # 5 blocks x 30
        # The session notes record that a nurse interrupted the tongue-pulling block and
        # that about three trials may need dropping. The indices are not knowable until the
        # trial table exists, so this stays empty rather than carrying a guess.
        "exclude_trials": [],
        "out_dir":        os.path.join(_MICROEPI_RAW, r"MicroEPI-G-03\task_FBM\data_MM"),
        "status":         "ready (needs time_range)",
        "notes":          "Experiment 11, 13/03/2025 pm, Jonathan & Justine present. "
                          "15:08-15:28, confirmed by the behavioural log's own filename "
                          "datetime 20250313-150809. Disturbed by nurse during the tongue "
                          "pulling block, ~3 trials may need excluding.",
    },
    # -------------------------------------------------------------------
    "G-05": {
        "pat_name":       "PAT_6684",
        "data_dir":       os.path.join(_MICROEPI_RAW, r"MicroEPI-G-05\task_FBM\MM"),
        "mat_files":      ["f0001_export_Labs_phmicrodown.mat",
                           "f0002_export_Labs_phmicrodown.mat"],
        "beh_dir":        os.path.join(_MICROEPI_RAW, r"MicroEPI-G-05\task_FBM\MM\raw"),
        "tsv_file":       "sub-microepi-g-05_task-motormapping_datetime-20250623-133155.tsv",
        "electrodes_tsv": os.path.join(_BIDS_ELEC_MICROEPI, r"sub-6684\ieeg\*_electrodes.tsv"),
        "trig":           "photodiode",
        "flip":           False,
        # PROVISIONAL, and the number that makes this patient work. f0001 is ~1169 s and
        # carries a clean cue train from ~469 s to ~1167 s; before 469 s sit two calibration
        # flashes ~40x the cue amplitude (6,400 against a 109-unit swing). Including them
        # puts the detection threshold 40x too high and yields 9 events instead of ~144.
        # This window is in the EXPORT's clock, not the TRC's - the TRC is only 878.5 s long.
        # BLACKROCK SESSION SECONDS, from Lora, read off the trigger QC figure. This
        # replaces (469, 1167), which came from the .mat export's clock and pointed at a
        # stretch of the Blackrock session that is flat - the two clocks differ by about
        # 230 s, which is exactly the kind of error that carrying one number between two
        # clocks produces.
        "time_range":     (700.0, 1420.0),
        # Per-block bounds, same clock. Given rather than inferred: the between-block gaps
        # are not reliably longer than a slow trial, so splitting on gap size fragmented a
        # 5-block session into 12. With bounds the assignment is exact and the trial counts
        # per block become a CHECK rather than an assumption.
        "block_bounds":   [[700, 850], [850, 990], [990, 1120], [1120, 1250], [1270, 1420]],
        "pd_norm_blocks": None,     # one window is enough here; the train is regular
        # MACRO CLOCK OFFSET:  micromed_t = blackrock_session_t + align_macro_offset_s
        # Measured, not assumed. Nine micro/macro channel pairs across four shafts
        # (AGm/CAGm/HAGm/HPGm against AG/CAG/HAG/HPG) cross-correlated at 1-40 Hz all put
        # part 1393's start at TRC t = 325.7 s, spread 0.7 s. Part 1393 begins 905.1 s into
        # the session, so the offset is 325.7 - 905.1 = -579.4 s.
        # Sanity check that it is the right sign and magnitude: the cue train at session
        # 700-1420 s maps to TRC 120.6-840.6 s, and the TRC is 878.5 s long - the whole
        # task fits inside the file with room at both ends, which a wrong offset would not.
        # Precision is about +-0.35 s: fine for cutting 6.2 s trials, too coarse to put a
        # spike raster against a cue. See the note in 20_mm_macro.py.
        "align_macro_offset_s": -579.4,
        # ...and that is ONLY right for part 1393. The parts are not contiguous (Central
        # leaves 6.34 s and 4.81 s between them here - header origins, sync cadence and a
        # trial missing from the photodiode in each gap all agree), so the concatenated
        # clock is off by the gap on every other part. 11_mm_sync.py measures one offset
        # per part from the sync pulses MKR4+ and the .nev both carry (residual ~3.5 ms),
        # seeded by this value on this part; 20_mm_macro.py uses the per-part table.
        "align_anchor_part": 1393,
        "bk_pd_channel":  "ainp1",
        "bk_pd_p2p":      143,      # measured - against a 6661 calibration flash, hence
                                    # time_range below; a 47x ratio inside one file
        "trc":            "EEG_2675077.TRC",
        "trc_marker":     ("expe motor mapping jon", None),
        "trc_block_s":    (4.03, None),
        "blackrock_dir":  os.path.join(_MICROEPI_RAW, r"MicroEPI-G-05\raw_blackrock\20250618-145815"),
        "blackrock":      ("20250618-145815-1387", "20250618-145815-1396"),
        "n_trials_expected": 100,   # 5 blocks x 20
        "exclude_trials": [],
        "out_dir":        os.path.join(_MICROEPI_RAW, r"MicroEPI-G-05\task_FBM\MM"),
        "status":         "ready",
        "notes":          "Experiment 15, 23/06/2025. 150 channels including IMG1-IMG18. "
                          "144 photodiode transitions in the window, median gap 3.11 s "
                          "against a design of 3.2 s baseline + 3.0 s stimulus - the "
                          "detector is toggling at both transitions, so edges are ~2x trials.",
    },
}

# The patients this pipeline loops over. G-04 and G-06 never ran the task.
patient_ids = ["G-01", "G-02", "G-03", "G-05"]

# The ones that can actually run right now. Kept separate rather than silently filtering
# patient_ids, so a blocked patient stays visible instead of disappearing from the loop.
patient_ids_ready = [p for p, c in MM_PRESETS.items()
                     if str(c.get("status", "")).startswith("ready")]

# ---------------------------------------------------------------------------
# signal processing - MACRO (Micromed), the ERSP side
# ---------------------------------------------------------------------------
# NO TIME WARPING. Every trial is the same length by design (a fixed 3.0 s stimulus after a
# ~3.2 s baseline), so there is nothing to warp to, and warping a fixed-length trial would
# only blur the onset. compute_ersp is therefore called with mode="RT", the same
# non-warped path notebook 150 uses.
mode = "RT"

# THE TRIAL, AS THE LOG DEFINES IT AND THE PHOTODIODE CONFIRMS:
#   a REST period of varying length  (log BaselineDuration 3.10-3.40 s, sd 0.107;
#                                     detected 3.04-4.81 s, median 3.22)
#   then a GO period of fixed length (log StimuliDuration exactly 3.00 s, sd 0.000;
#                                     detected median 3.00, sd 0.025)
# t = 0 in every epoch is GO ONSET, so everything before 0 is rest.
time_window = (-2.0, 5.0)         # covers the tail of rest, all of GO, and the return to rest

# BASELINE: 800 ms of rest, ending 200 ms before GO. The buffer matters - an epoch boundary
# placed right at the cue lets the movement's own onset leak backwards through the STFT
# window (128 samples = 125 ms) and into the very baseline it is being compared against,
# which shrinks the measured response. Ending at -0.2 s keeps the whole analysis window
# clear of it. The shortest rest observed is 3.04 s, so -1.0..-0.2 sits two seconds inside
# rest even on the tightest trial and can never catch the previous trial's movement.
baseline_w = (-1.0, -0.2)
baseline_calc_w = (-1.0, -0.2)

# THE ERSP DISPLAY AXIS: first 20% baseline, remaining 80% GO, drawn square.
# GO is a fixed 3.0 s, so fixing it at 80% of the width fixes the baseline shown at
# 3.0 * 20/80 = 0.75 s, and t = 0 lands exactly one fifth of the way across every panel.
# This is a DISPLAY choice and nothing else: the dB values are still computed against
# baseline_calc_w (-1.0..-0.2 s), which extends slightly further back than the panel
# shows, so the statistics do not change when the framing does.
ersp_baseline_frac = 0.20
ersp_display_window = (-design_stim_s * ersp_baseline_frac / (1.0 - ersp_baseline_frac),
                       design_stim_s)

# Native sampling rate per patient, never resampled to a common grid: the two systems are
# 2048 Hz (Micromed) and 30 kHz (Blackrock), and a spike analysis cannot survive a
# downsample to 1 kHz. compute_ersp resamples internally for the STFT only.
fs_resample_hz = None             # None = keep the recording's own rate

# STFT. fmax 400 rather than LM's 500 because the question here is muscle contamination,
# which lives in the broadband above 70 Hz, and 400 Hz is where the band tables stop.
nperseg = 128
nfft = nperseg * 2
noverlap = int(0.85 * nperseg)
fmax = 400.0
ersp_vlim = 6.0                   # +/- dB on the ERSP colour scale

# High gamma, the band the muscle question is actually about.
hg_band = (70.0, 150.0)
hg_smooth_ms = 25
hg_vmin, hg_vmax = -6.0, 6.0

# Broadband, for the mouth-vs-foot contrast: muscle is flat and wide, not banded.
broadband = (70.0, 150.0)
broadband_wide = (70.0, 400.0)

# PSD, raw and cleaned, per condition block.
psd_fmax = 400.0
psd_dpi = 600
mains_base = 50.0
notch_scope = "block"             # notch each condition block on its own spectrum
notch_block_pad_s = 10.0
peak_z_thresh = 3.0               # a harmonic is notched only if its peak clears this

# Referencing, as for LM.
reref_type = "WM"
wm_ref_method = "mean"
wm_min_contacts = 3

# WM CONTACTS TO KEEP OUT OF THE REFERENCE, per patient. The anatomy table's 0.97 cut is
# the LM rule and a good one, but a contact a millimetre from grey matter can pass it and
# still see cortex; a reference built from it subtracts a real response from every other
# channel. Decide from the control run's WM_review contact sheets (--reref none), then list
# the offenders here. This is merged into the exclusion list LM's own
# apply_wm_reference_with_exclusions takes, so the mechanism is LM's; only the list is
# MM-specific. A contact that is not WM here is not WM for LM either - the long-term home
# for it is bad_channels_manual in the LM config - but that changes LM runs, so test the
# decision here first.
wm_exclude = {
    "G-05": [],
}

# Trial rejection, off by default, same knobs as the LM pipeline.
ersp_trial_reject = {}
ersp_trial_reject_z = {}
ersp_trial_reject_hg_mad = {}
ersp_trial_reject_hg_z = {}

# ---------------------------------------------------------------------------
# signal processing - MICRO (Blackrock), the LFP side
# ---------------------------------------------------------------------------
# Decimation target for the cached micro LFP. Nyquist 2500 Hz, which is what lets the
# notch reach 2000 and the ERSP reach 1000. Raising this raises the cache linearly.
micro_lfp_fs = 5000.0

# Clean further up than you look: harmonics above the display limit still fold energy into
# the estimate through the STFT window, so the notch runs to 2000 Hz while the ERSP is
# drawn to 1000.
micro_notch_fmax = 2000.0
micro_ersp_fmax = 1000.0
micro_psd_fmax = 1200.0

# Referencing for the micro contacts. "none" is as recorded.
#   "first"      subtract the first contact of the same shaft (AGm1 for all AGm)
#   "shaft_mean" subtract the mean of that shaft's contacts
# Both are offered because they fail differently - see the module docstring. The reference
# contact itself becomes identically zero under "first" and is dropped from the figures
# rather than drawn as a flat line pretending to be data.
micro_reref = "shaft_mean"

# The Blackrock analog inputs (ainp1 = photodiode, ainp2/3 whatever was plugged in - in
# G-05 nothing: they are identical to each other and white) go through the same cache,
# notch, epoching, HG and ERSP as the micro contacts. They are not on a shaft, so they are
# never a reference and never re-referenced. What they show is the task as seen by the
# same amplifier with no tissue attached; a response there is not neural and not muscle.
micro_include_ainp = True

# The cache holds the task window only, with this much margin either side. At 5000 Hz the
# whole session would be about 1.5 GB per patient; the task is half of it and nothing
# outside the window is ever epoched.
micro_cache_pad_s = 30.0

# ---------------------------------------------------------------------------
# signal processing - MICRO (Blackrock), the spike side
# ---------------------------------------------------------------------------
# "Motor hits" = spiking that follows the cue. Detection is on the 30 kHz broadband after a
# bandpass, threshold as a multiple of the robust noise estimate rather than the SD, since
# the SD of a channel that is spiking is inflated by the spikes themselves.
# WHERE spikes are detected. A Blackrock session can run for weeks; the task is twenty
# minutes of it. The window is derived from the cue train - first cue to last cue, padded -
# so nothing here goes stale when a patient is re-extracted.
spike_window_from = "cue_train"       # "cue_train" | an explicit (t0, t1) in session seconds
spike_window_pad_s = 30.0

# THE .nev DIGITAL EVENTS ARE SYNCHRONISATION, NOT TRIALS. They arrive on a regular ~5.78 s
# cadence carrying a 4-byte word, which is exactly what a trial code looks like, and they
# are not one. Trial onsets come from the photodiode cue train on ainp1. Any digital stream
# with that regular cadence belongs to the sync hardware and must be ignored.
nev_digital_is_sync = True

spike_band = (300.0, 3000.0)
spike_thresh_k = 4.0              # k in  thr = k * median(|x|) / 0.6745
spike_polarity = "neg"            # "neg" | "pos" | "both"
spike_refractory_ms = 1.0
spike_window_ms = (-1.0, 2.0)     # waveform cut around each detection

# Firing-rate comparison per condition, the micro equivalent of the ERSP contrast.
rate_bin_ms = 50
rate_smooth_ms = 100
rate_baseline_w = (-2.0, -0.2)

__all__ = [
    "block_name", "conditions_expected", "COND_ALIAS", "COND_SHORT",
    "CONTRAST_PRIMARY", "CONTRAST_CONTROL", "MM_PRESETS",
    "patient_ids", "patient_ids_ready",
]
