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

Box geometry:
  t_bins = (t_lo, t_hi)  -> TIME, 1-based inclusive, PER BLOCK (grid-dependent).
  f_hz   = (f_lo, f_hi)  -> FREQUENCY in Hz (grid-INDEPENDENT). The engine (lf_pool)
           converts Hz -> rows per grid:
             full : linear 0-500 Hz over 129 rows  (row = round(hz/500*128)+1)
             ds   : the 15 bands overlapping the Hz range
  Frequency windows are written ONCE in Hz and work on both grids; only the time
  windows differ between grids.

Grid specs (time only):
  ds   : 30 time bins per block (stim 1-15, resp 16-30)
  full : 300 time bins per block (stim 1-150, resp 151-300)
"""

BLOCK_ORDER = ["audio", "picture", "reading"]

# Frequency bands in Hz — grid-independent; lf_pool converts Hz -> rows per grid.
HGA_HZ          = (70, 170)   # high-gamma activity
ALPHA_BETA_HZ   = (8,  30)    # alpha + beta (movement / engagement ERD)
THETA_HZ        = (4,  8)     # theta (semantic retrieval)
BROADBAND_HF_HZ = (70, 400)   # broadband high-freq (HGA + HFO) — negative networks
LOW_F_HZ        = (0,  40)    # low frequency (delta-beta) — low-f negative network
AUD_SUPP_HZ     = (20, 45)    # low-gamma/beta band suppressed while hearing


def _roles(hga, alpha_beta, theta, stim, resp, pre_resp, motor_win, nt):
    """
    Build role list for one grid.
    hga, alpha_beta, theta = (f_lo_hz, f_hi_hz) tuples IN HZ (grid-independent)
    stim, resp, pre_resp   = (t_lo, t_hi) time-bin tuples (grid-dependent)
    motor_win              = (t_lo, t_hi) for the motor role (60-90% of the block)
    nt                     = n_time_per_block (30 ds / 300 full) — for pct() windows
    pre_resp covers the last 10% of the response window (word search window)
    """
    f   = list(hga)
    fab = list(alpha_beta)
    fth = list(theta)
    S   = list(stim)
    R   = list(resp)
    PR  = list(pre_resp)
    MW  = list(motor_win)

    def pct(lo, hi):
        """[lo%, hi%] of the per-block time range -> 1-based inclusive bins."""
        return [max(1, round(lo / 100.0 * nt)), min(nt, round(hi / 100.0 * nt))]

    def stim_first(p):
        """first p% of the STIMULUS window -> 1-based inclusive bins."""
        s0, s1 = S
        return [s0, min(s1, s0 - 1 + max(1, round(p / 100.0 * (s1 - s0 + 1))))]

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
                {"block": "audio",   "t_bins": S,  "f_hz": f, "sign": "pos"},  # hears stimulus
                {"block": "audio",   "t_bins": R,  "f_hz": f, "sign": "pos"},  # hears self
                {"block": "picture", "t_bins": R,  "f_hz": f, "sign": "pos"},  # hears self
                {"block": "reading", "t_bins": R,  "f_hz": f, "sign": "pos"},  # hears self
                # MUST NOT be on
                {"block": "picture", "t_bins": S,  "f_hz": f, "sign": "zero"}, # not visual input
                {"block": "reading", "t_bins": S,  "f_hz": f, "sign": "zero"}, # not visual input
            ],
        },

        # ── 2. VISUAL ──────────────────────────────────────────────────
        {
            "role": "visual",
            "description": (
                "Visual cortex: fast onset HGA to the VISUAL stimulus — the first 70% "
                "of the picture stim window and only the first 10% of the reading stim "
                "window (text drives a briefer early visual response). Silent during "
                "audio stimulus and all response windows. "
                "Citation: Crone et al. 1998 — gamma ERS is somatotopically "
                "organized and spatially focal; "
                "Ray & Maunsell 2011 — HGA tracks local spiking activity "
                "with high spatial specificity."
            ),
            "color": "#2ca02c",
            "boxes": [
                # MUST be on — early visual onset only (not the whole stim window)
                {"block": "picture", "t_bins": stim_first(70), "f_hz": f, "sign": "pos"},  # first 70% of picture stim
                {"block": "reading", "t_bins": stim_first(10), "f_hz": f, "sign": "pos"},  # first 10% of reading stim
                # MUST NOT be on
                {"block": "audio",   "t_bins": S,  "f_hz": f, "sign": "zero"}, # no auditory input
                {"block": "audio",   "t_bins": R,  "f_hz": f, "sign": "zero"}, # no response
                {"block": "picture", "t_bins": R,  "f_hz": f, "sign": "zero"}, # no response
                {"block": "reading", "t_bins": R,  "f_hz": f, "sign": "zero"}, # no response
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
                {"block": "audio",   "t_bins": MW, "f_hz": f,   "sign": "pos"},
                {"block": "picture", "t_bins": MW, "f_hz": f,   "sign": "pos"},
                {"block": "reading", "t_bins": MW, "f_hz": f,   "sign": "pos"},
                # beta AT MOST ZERO — movement suppression (neg or flat OK, never activated)
                {"block": "audio",   "t_bins": MW, "f_hz": fab, "sign": "nonpos"},
                {"block": "picture", "t_bins": MW, "f_hz": fab, "sign": "nonpos"},
                {"block": "reading", "t_bins": MW, "f_hz": fab, "sign": "nonpos"},
                # MUST NOT be on — silent during all stimuli
                {"block": "audio",   "t_bins": S,  "f_hz": f,   "sign": "zero"},
                {"block": "picture", "t_bins": S,  "f_hz": f,   "sign": "zero"},
                {"block": "reading", "t_bins": S,  "f_hz": f,   "sign": "zero"},
            ],
        },

        # ── 7. WORD SEARCH / LEXICAL RETRIEVAL ────────────────────────
        {
            "role": "word_search",
            "description": (
                "Lexical retrieval: brief HGA burst at the END of the STIMULUS window "
                "(40-50% of the block, the pre-articulation period just before response "
                "onset) across all three conditions. Quiet at stimulus ONSET (first half "
                "of stim) — distinguishes it from a fast sensory-onset response. "
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
                # MUST be on — brief HGA burst at the END of the stimulus, all conditions
                {"block": "audio",   "t_bins": pct(40, 50),    "f_hz": f, "sign": "pos"},
                {"block": "picture", "t_bins": pct(40, 50),    "f_hz": f, "sign": "pos"},
                {"block": "reading", "t_bins": pct(40, 50),    "f_hz": f, "sign": "pos"},
                # MUST NOT be on — quiet at stimulus ONSET (not a sensory-onset cell)
                {"block": "audio",   "t_bins": stim_first(50), "f_hz": f, "sign": "zero"},
                {"block": "picture", "t_bins": stim_first(50), "f_hz": f, "sign": "zero"},
                {"block": "reading", "t_bins": stim_first(50), "f_hz": f, "sign": "zero"},
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
                {"block": "audio",   "t_bins": S, "f_hz": fab, "sign": "neg"},
                {"block": "picture", "t_bins": S, "f_hz": fab, "sign": "neg"},
                {"block": "reading", "t_bins": S, "f_hz": fab, "sign": "neg"},
            ],
        },

        # ── 9. AUDITORY SUPPRESSION (20-45 Hz while hearing) ───────────
        {
            "role": "auditory_suppression",
            "description": (
                "Low-gamma/beta (20-45 Hz) suppression whenever the patient is actively "
                "hearing — the audio stimulus (hearing the prompt) and all three response "
                "windows (hearing own voice). Negative activity is particularly strong in "
                "the audio condition. Separate marker from the HGA-based auditory role. "
                "Citation: Crone et al. 1998 — beta/low-gamma ERD accompanies auditory "
                "and speech processing; Ray & Maunsell 2011 — beta suppression is a "
                "distinct marker from broadband HGA."
            ),
            "color": "#17becf",
            "boxes": [
                {"block": "audio",   "t_bins": S, "f_hz": list(AUD_SUPP_HZ), "sign": "neg"},  # hears prompt
                {"block": "audio",   "t_bins": R, "f_hz": list(AUD_SUPP_HZ), "sign": "neg"},  # hears self
                {"block": "picture", "t_bins": R, "f_hz": list(AUD_SUPP_HZ), "sign": "neg"},  # hears self
                {"block": "reading", "t_bins": R, "f_hz": list(AUD_SUPP_HZ), "sign": "neg"},  # hears self
            ],
        },

        # ── 10. NETWORK NEGATIVE 1 (broadband HF suppression + late silence) ──
        {
            "role": "network_negative_1",
            "description": (
                "Broadband high-frequency (70-400 Hz) suppression over the early-to-mid "
                "trial (10-60% of the block) in all three conditions, returning to baseline "
                "(zero) late (80-100% of the block). A task-negative / deactivation network "
                "marker. Citation: Raichle 2015 — task-negative deactivation; "
                "Ossandon et al. 2011 — sustained HGA decreases (deactivation) in "
                "default-mode regions during effortful tasks."
            ),
            "color": "#2b8cbe",
            "boxes": [
                # MUST suppress (broadband HF) early-mid in all conditions
                {"block": "audio",   "t_bins": pct(10, 60), "f_hz": list(BROADBAND_HF_HZ), "sign": "neg"},
                {"block": "picture", "t_bins": pct(10, 60), "f_hz": list(BROADBAND_HF_HZ), "sign": "neg"},
                {"block": "reading", "t_bins": pct(10, 60), "f_hz": list(BROADBAND_HF_HZ), "sign": "neg"},
                # MUST return to ~zero late
                {"block": "audio",   "t_bins": pct(80, 100), "f_hz": list(BROADBAND_HF_HZ), "sign": "zero"},
                {"block": "picture", "t_bins": pct(80, 100), "f_hz": list(BROADBAND_HF_HZ), "sign": "zero"},
                {"block": "reading", "t_bins": pct(80, 100), "f_hz": list(BROADBAND_HF_HZ), "sign": "zero"},
            ],
        },

        # ── 11. NETWORK NEGATIVE 2 (sustained broadband HF suppression) ──
        {
            "role": "network_negative_2",
            "description": (
                "Sustained broadband high-frequency (70-400 Hz) suppression over most of "
                "the block (10-80%) in all three conditions, with NO late-silence "
                "constraint. A broader / longer-lasting deactivation than network_negative_1. "
                "Citation: Raichle 2015 — task-negative deactivation; Ossandon et al. 2011."
            ),
            "color": "#045a8d",
            "boxes": [
                {"block": "audio",   "t_bins": pct(10, 80), "f_hz": list(BROADBAND_HF_HZ), "sign": "neg"},
                {"block": "picture", "t_bins": pct(10, 80), "f_hz": list(BROADBAND_HF_HZ), "sign": "neg"},
                {"block": "reading", "t_bins": pct(10, 80), "f_hz": list(BROADBAND_HF_HZ), "sign": "neg"},
            ],
        },

        # ── 12. LOW-F NEGATIVE NETWORK (0-40 Hz mid-trial suppression) ──
        {
            "role": "low_f_negative_network",
            "description": (
                "Low-frequency (0-40 Hz) suppression in the middle of the block "
                "(30-70% of the time, spanning the stim->response transition) in all "
                "three conditions. Low-frequency ERD marker distinct from the broadband "
                "HF negative networks. Citation: Crone et al. 1998 — alpha/beta ERD "
                "during active processing; Jia & Kohn 2011 — low-frequency power "
                "suppression during network engagement."
            ),
            "color": "#74a9cf",
            "boxes": [
                {"block": "audio",   "t_bins": pct(30, 70), "f_hz": list(LOW_F_HZ), "sign": "neg"},
                {"block": "picture", "t_bins": pct(30, 70), "f_hz": list(LOW_F_HZ), "sign": "neg"},
                {"block": "reading", "t_bins": pct(30, 70), "f_hz": list(LOW_F_HZ), "sign": "neg"},
            ],
        },

        # ── 13. STIMULUS RESPONSIVE (umbrella — HGA in ANY stim window) ──
        {
            "role": "stimulus_responsive",
            "match": "any",          # disjunction: matches if >=1 box expresses
            "description": (
                "General stimulus-driven HGA: high-gamma activation in the stimulus "
                "window of AT LEAST ONE condition (audio, picture, or reading). No "
                "selectivity constraint — an UMBRELLA tag that co-occurs with the "
                "specific sensory/language roles (an 'auditory' contact is therefore "
                "also 'stimulus_responsive'), giving a general+specific hierarchy "
                "rather than one exclusive label."
            ),
            "color": "#aec7e8",
            "boxes": [
                {"block": "audio",   "t_bins": S, "f_hz": f, "sign": "pos"},
                {"block": "picture", "t_bins": S, "f_hz": f, "sign": "pos"},
                {"block": "reading", "t_bins": S, "f_hz": f, "sign": "pos"},
            ],
        },

        # ── 14. RESPONSE ACTIVE (umbrella — HGA in ANY response window) ──
        {
            "role": "response_active",
            "match": "any",          # disjunction: matches if >=1 box expresses
            "description": (
                "General response/production-driven HGA: high-gamma activation in the "
                "response window of AT LEAST ONE condition. UMBRELLA tag (no selectivity) "
                "that co-occurs with motor / language / word_search etc."
            ),
            "color": "#ffbb78",
            "boxes": [
                {"block": "audio",   "t_bins": R, "f_hz": f, "sign": "pos"},
                {"block": "picture", "t_bins": R, "f_hz": f, "sign": "pos"},
                {"block": "reading", "t_bins": R, "f_hz": f, "sign": "pos"},
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
            hga        = HGA_HZ,
            alpha_beta = ALPHA_BETA_HZ,
            theta      = THETA_HZ,
            stim       = (1,  15),
            resp       = (16, 30),
            pre_resp   = (28, 30),
            motor_win  = (18, 27),     # 60-90% of the 30-bin block
            nt         = 30,
        ),
    },
    "full": {
        "n_time_per_block": 300,
        "n_freq": 129,
        "stim_bins":     [1,   150],
        "resp_bins":     [151, 300],
        "pre_resp_bins": [271, 300],
        "roles": _roles(
            hga        = HGA_HZ,
            alpha_beta = ALPHA_BETA_HZ,
            theta      = THETA_HZ,
            stim       = (1,  150),
            resp       = (151, 300),
            pre_resp   = (271, 300),
            motor_win  = (180, 270),   # 60-90% of the 300-bin block
            nt         = 300,
        ),
    },
}