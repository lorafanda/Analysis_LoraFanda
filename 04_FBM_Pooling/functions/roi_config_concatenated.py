"""
roi_config_concatenated.py

Functional-role conjunction templates over the CONCATENATED
[audio | picture | reading] ERSP.

Each role is defined as a set of boxes with expected signs.
A contact matches a role only if ALL boxes are satisfied (strict AND).
Boxes with sign="pos" must show positive mean ERSP above threshold.
Boxes with sign="neg" must show negative mean ERSP below threshold.
Boxes with sign="zero" must show mean ERSP close to zero (no response).

Block coordinates are PER-BLOCK (1-based, inclusive).
The engine (lf_pool) adds the block offset when slicing the
concatenated map. Block order: audio=0, picture=1, reading=2.

Grid specs:
  ds   : 30 time bins per block (stim 1-15, resp 16-30); 15 freq bands
  full : 300 time bins per block (stim 1-150, resp 151-300); 129 freq rows

Frequency band mappings:
  ds   : HGA=[8,10] (~70-170Hz), alpha_beta=[3,5] (~8-30Hz),
         theta=[2,2] (~4-8Hz), narrowband_gamma=[6,7] (~30-70Hz)
  full : HGA=[19,44] (~70-170Hz), alpha_beta=[3,6] (~8-30Hz),
         theta=[2,3] (~4-8Hz), narrowband_gamma=[9,19] (~30-70Hz)
"""

BLOCK_ORDER = ["audio", "picture", "reading"]


def _roles(hga, alpha_beta, theta, stim, resp, pre_resp, motor_win):
    """
    Build role list for one grid.
    hga, alpha_beta, theta = (f_lo, f_hi) tuples
    stim, resp, pre_resp   = (t_lo, t_hi) tuples
    motor_win              = (t_lo, t_hi) for the motor role (60-90% of the block)
    pre_resp covers the last 10% of the response window (word search window)
    """
    f   = list(hga)
    fab = list(alpha_beta)
    fth = list(theta)
    S   = list(stim)
    R   = list(resp)
    PR  = list(pre_resp)
    MW  = list(motor_win)

    return [

        # ── 1. AUDITORY ────────────────────────────────────────────────
        {
            "role": "auditory",
            "description": (
                "Auditory cortex: responds to the audio STIMULUS and to "
                "own voice in all three response windows. "
                "Discriminated from motor by the audio stim box. "
                "Discriminated from visual by absence of picture/reading stim boxes. "
                "Citation: Trebuchon et al. 2020 — auditory word-related HGA flows "
                "from pSTG encoding acoustic properties; "
                "Kitazawa & Asano et al. 2025 — bilateral STG HGA augmentation "
                "within 100ms of first phrase onset."
            ),
            "color": "#1f77b4",
            "boxes": [
                # MUST be on
                {"block": "audio",   "t_bins": S,  "f_rows": f, "sign": "pos"},  # hears stimulus
                {"block": "audio",   "t_bins": R,  "f_rows": f, "sign": "pos"},  # hears self
                {"block": "picture", "t_bins": R,  "f_rows": f, "sign": "pos"},  # hears self
                {"block": "reading", "t_bins": R,  "f_rows": f, "sign": "pos"},  # hears self
                # MUST NOT be on
                {"block": "picture", "t_bins": S,  "f_rows": f, "sign": "zero"}, # not visual input
                {"block": "reading", "t_bins": S,  "f_rows": f, "sign": "zero"}, # not visual input
            ],
        },

        # ── 2. VISUAL ──────────────────────────────────────────────────
        {
            "role": "visual",
            "description": (
                "Visual cortex: responds to picture and reading STIMULI only. "
                "Silent during audio stimulus and all response windows. "
                "Citation: Crone et al. 1998 — gamma ERS is somatotopically "
                "organized and spatially focal; "
                "Ray & Maunsell 2011 — HGA tracks local spiking activity "
                "with high spatial specificity."
            ),
            "color": "#2ca02c",
            "boxes": [
                # MUST be on
                {"block": "picture", "t_bins": S,  "f_rows": f, "sign": "pos"},
                {"block": "reading", "t_bins": S,  "f_rows": f, "sign": "pos"},
                # MUST NOT be on
                {"block": "audio",   "t_bins": S,  "f_rows": f, "sign": "zero"}, # no auditory input
                {"block": "audio",   "t_bins": R,  "f_rows": f, "sign": "zero"}, # no response
                {"block": "picture", "t_bins": R,  "f_rows": f, "sign": "zero"}, # no response
                {"block": "reading", "t_bins": R,  "f_rows": f, "sign": "zero"}, # no response
            ],
        },

        # ── 3. MOTOR / SPEECH ──────────────────────────────────────────
        {
            "role": "motor",
            "description": (
                "Speech-motor cortex: HGA during articulation at 60-90% of the block "
                "(late response) in all three conditions, with beta AT MOST ZERO "
                "(movement ERD — suppression or flat, never activated). "
                "Silent during all stimulus windows — discriminates from sensory. "
                "Citation: Crone et al. 1998 — high-gamma ERS in peri-rolandic "
                "cortex is somatotopically organized during motor tasks, with "
                "simultaneous alpha/beta ERD; "
                "Trebuchon et al. 2020 — visual naming elicits HGA in "
                "peri-rolandic and premotor regions."
            ),
            "color": "#d62728",
            "boxes": [
                # MUST be on — HGA articulation, 60-90% of the block
                {"block": "audio",   "t_bins": MW, "f_rows": f,   "sign": "pos"},
                {"block": "picture", "t_bins": MW, "f_rows": f,   "sign": "pos"},
                {"block": "reading", "t_bins": MW, "f_rows": f,   "sign": "pos"},
                # beta AT MOST ZERO — movement suppression (neg or flat OK, never activated)
                {"block": "audio",   "t_bins": MW, "f_rows": fab, "sign": "nonpos"},
                {"block": "picture", "t_bins": MW, "f_rows": fab, "sign": "nonpos"},
                {"block": "reading", "t_bins": MW, "f_rows": fab, "sign": "nonpos"},
                # MUST NOT be on — silent during all stimuli
                {"block": "audio",   "t_bins": S,  "f_rows": f,   "sign": "zero"},
                {"block": "picture", "t_bins": S,  "f_rows": f,   "sign": "zero"},
                {"block": "reading", "t_bins": S,  "f_rows": f,   "sign": "zero"},
            ],
        },

        # ── 4. MULTIMODAL SENSORY ──────────────────────────────────────
        {
            "role": "multimodal_sensory",
            "description": (
                "Heteromodal cortex: HGA to the STIMULUS in all three conditions "
                "regardless of modality. Silent during all response windows. "
                "Citation: Lachaux et al. 2012 — HGA reveals participation of "
                "local neural ensembles across cognitive domains; "
                "Parvizi & Kastner 2018 — HGA is the most reliable marker of "
                "local cortical activation across motor, sensory, and language domains."
            ),
            "color": "#9467bd",
            "boxes": [
                # MUST be on
                {"block": "audio",   "t_bins": S,  "f_rows": f, "sign": "pos"},
                {"block": "picture", "t_bins": S,  "f_rows": f, "sign": "pos"},
                {"block": "reading", "t_bins": S,  "f_rows": f, "sign": "pos"},
                # MUST NOT be on
                {"block": "audio",   "t_bins": R,  "f_rows": f, "sign": "zero"},
                {"block": "picture", "t_bins": R,  "f_rows": f, "sign": "zero"},
                {"block": "reading", "t_bins": R,  "f_rows": f, "sign": "zero"},
            ],
        },

        # ── 5. LANGUAGE (full network) ─────────────────────────────────
        {
            "role": "language",
            "description": (
                "Language network: HGA active during both stimulus AND response "
                "in all three conditions. Captures the full perception-to-production "
                "chain. More boxes than motor or auditory alone — most specific "
                "language role. "
                "Citation: Trebuchon et al. 2020 — no single modality is adequate; "
                "language is distributed across stimulus and response windows; "
                "Kitazawa & Asano et al. 2025 — left intra-hemispheric functional "
                "connectivity enhancement greatest prior to verbal response across "
                "all conditions."
            ),
            "color": "#ff7f0e",
            "boxes": [
                # MUST be on — stimulus and response in all three conditions
                {"block": "audio",   "t_bins": S,  "f_rows": f, "sign": "pos"},
                {"block": "audio",   "t_bins": R,  "f_rows": f, "sign": "pos"},
                {"block": "picture", "t_bins": S,  "f_rows": f, "sign": "pos"},
                {"block": "picture", "t_bins": R,  "f_rows": f, "sign": "pos"},
                {"block": "reading", "t_bins": S,  "f_rows": f, "sign": "pos"},
                {"block": "reading", "t_bins": R,  "f_rows": f, "sign": "pos"},
            ],
        },

        # ── 6. CONTEXT EXTRACTION ──────────────────────────────────────
        {
            "role": "context_extraction",
            "description": (
                "Sentence-level semantic integration: fires during complex sentence "
                "STIMULI (audio and reading) but NOT during picture viewing. "
                "Picture is a single object — no multi-word integration required. "
                "Audio and reading both require sequential word integration and "
                "semantic context building across time. "
                "Uses both HGA and theta — theta specifically indexes semantic "
                "retrieval during sentence processing. "
                "Citation: Nature Comms 2023 (semantic integration iEEG, 58 patients) "
                "— broadband HGA (70-150Hz) maps sentence comprehension across "
                "language-dominant cortex; sentences triggering inferential semantics "
                "recruit networks that picture naming does not; "
                "bioRxiv 2025 (decoding semantics from sEEG) — semantic information "
                "encoded from theta to gamma; lower frequencies show higher decoding "
                "accuracy for semantic retrieval; "
                "Johnson et al. 2020 — theta synchronization linked to memory "
                "encoding and successful recall."
            ),
            "color": "#8c564b",
            "boxes": [
                # MUST be on — HGA during audio and reading stimulus
                {"block": "audio",   "t_bins": S,  "f_rows": f,   "sign": "pos"},
                {"block": "reading", "t_bins": S,  "f_rows": f,   "sign": "pos"},
                # MUST be on — theta during audio and reading stimulus
                {"block": "audio",   "t_bins": S,  "f_rows": fth, "sign": "pos"},
                {"block": "reading", "t_bins": S,  "f_rows": fth, "sign": "pos"},
                # MUST NOT be on — picture stimulus should be silent or minimal
                {"block": "picture", "t_bins": S,  "f_rows": f,   "sign": "zero"},
            ],
        },

        # ── 7. WORD SEARCH / LEXICAL RETRIEVAL ────────────────────────
        {
            "role": "word_search",
            "description": (
                "Lexical retrieval: brief HGA burst in the late pre-response window "
                "(last 10% of response period, bins ~27-30 in ds / ~270-300 in full) "
                "across all three conditions. "
                "Absent or minimal during stimulus and early response. "
                "Reflects the final lexical selection and phonological encoding "
                "step before articulation. "
                "Citation: Sahin et al. 2009 (via Llorens et al. 2011 review) — "
                "iEEG Broca's area peaks at ~200ms (lexical), ~320ms (grammatical), "
                "~450ms (articulatory) post-stimulus; "
                "PNAS 2009 (overt speech word retrieval) — lexical retrieval starts "
                "~200ms post picture onset, unfolds for 180ms; "
                "Communications Biology 2025 (sEEG speech production) — "
                "IFG gamma and high-gamma onset at 200ms; pSTG activates just "
                "before articulation for phonological code retrieval; "
                "Kitazawa & Asano et al. 2025 — left pIFG HGA peaks 350-400ms "
                "before response onset."
            ),
            "color": "#e377c2",
            "boxes": [
                # MUST be on — brief HGA burst just before response in all conditions
                {"block": "audio",   "t_bins": PR, "f_rows": f, "sign": "pos"},
                {"block": "picture", "t_bins": PR, "f_rows": f, "sign": "pos"},
                {"block": "reading", "t_bins": PR, "f_rows": f, "sign": "pos"},
                # MUST NOT be on — stimulus windows should be quiet for this role
                {"block": "audio",   "t_bins": S,  "f_rows": f, "sign": "zero"},
                {"block": "picture", "t_bins": S,  "f_rows": f, "sign": "zero"},
                {"block": "reading", "t_bins": S,  "f_rows": f, "sign": "zero"},
            ],
        },

        # ── 8. ALPHA/BETA SUPPRESSION ──────────────────────────────────
        {
            "role": "alpha_beta_suppression",
            "description": (
                "Cortical engagement marker: alpha and beta suppression (ERD) "
                "during stimulus in all three conditions. "
                "No constraint on HGA — this role captures electrodes that show "
                "clear engagement via ERD without necessarily reaching HGA threshold. "
                "This is the RQ1 proof-of-concept role — electrodes matching this "
                "but NOT matching any HGA-based role would be missed by a pure "
                "high-gamma filter. "
                "Citation: Crone et al. 1998 — gamma ERS and alpha/beta ERD occur "
                "simultaneously and carry distinct functional information; "
                "Jia & Kohn 2011 — lower frequency power suppressed while higher "
                "frequency power increases during network activation; "
                "Ray & Maunsell 2011 — beta suppression is a separate marker "
                "from broadband HGA with different neural generator."
            ),
            "color": "#bcbd22",
            "boxes": [
                # MUST suppress during all three stimulus windows
                {"block": "audio",   "t_bins": S, "f_rows": fab, "sign": "neg"},
                {"block": "picture", "t_bins": S, "f_rows": fab, "sign": "neg"},
                {"block": "reading", "t_bins": S, "f_rows": fab, "sign": "neg"},
            ],
        },

    ]


# ── Pre-response window calculation ────────────────────────────────────────
# "word search" = last 10% of response window
# ds  : resp = [16,30], last 10% = bins [28,30]  (3 bins out of 15)
# full: resp = [151,300], last 10% = bins [271,300] (30 bins out of 150)
# Approximately 45-50% to 50% of the full trial in TN-normalized space

ERSP_POOLING_ROLES = {
    "block_order": BLOCK_ORDER,
    "ds": {
        "n_time_per_block": 30,
        "n_freq": 15,
        "stim_bins":     [1,  15],
        "resp_bins":     [16, 30],
        "pre_resp_bins": [28, 30],
        "roles": _roles(
            hga        = (8,  10),
            alpha_beta = (3,  5),
            theta      = (2,  2),
            stim       = (1,  15),
            resp       = (16, 30),
            pre_resp   = (28, 30),
            motor_win  = (18, 27),     # 60-90% of the 30-bin block
        ),
    },
    "full": {
        "n_time_per_block": 300,
        "n_freq": 129,
        "stim_bins":     [1,   150],
        "resp_bins":     [151, 300],
        "pre_resp_bins": [271, 300],
        "roles": _roles(
            hga        = (19, 44),
            alpha_beta = (3,  6),
            theta      = (2,  3),
            stim       = (1,  150),
            resp       = (151, 300),
            pre_resp   = (271, 300),
            motor_win  = (180, 270),   # 60-90% of the 300-bin block
        ),
    },
}