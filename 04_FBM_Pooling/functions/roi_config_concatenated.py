"""
roi_config_layered.py

RESTRUCTURED functional-role templates over the CONCATENATED
[audio | picture | reading] ERSP.

Design change vs. roi_config_concatenated.py
--------------------------------------------
The old flat list entangled three orthogonal axes (process / band / sign).
This version factors them into LAYERS, tagged via the "layer" key on each
role so downstream (lf_pool / cluster cards / website) can treat them as
separate label namespaces:

  layer 1  FUNCTIONAL roles — one per anatomical/process hypothesis, each
           carrying its FULL multiband fingerprint (HGA + beta ERD bundled,
           the way `motor` already did). No more X / X_suppression splits.
  layer 2  SPECTRAL tags   — orthogonal single-signature markers (HGA pos,
           beta ERD, low-f ERD, theta tracking). A contact that fires a
           layer-2 ERD tag but matches NO layer-1 HGA role is the
           "missed-by-HGA-only-filter" category (RQ1), expressed directly.
  layer 3  UMBRELLA tags   — coarse disjunctions (stimulus / response).

The engine, box semantics (pos/neg/zero/nonpos), and pct() helper are
UNCHANGED. Only the role list is reorganized + extended. The "layer" key is
ignored by the matching engine; it is metadata for grouping the outputs.

Windows are warped (TN). 50% = GO-cue. Production windows derive from the
single VOICE_ONSET_PCT assumption (replace with subset-measured ratio).
Block order: audio=0, picture=1, reading=2.
"""

BLOCK_ORDER = ["audio", "picture", "reading"]

# Frequency bands in Hz — grid-independent; lf_pool converts Hz -> rows per grid.
HGA_HZ   = (70, 150)   # high-gamma activation / suppression (deactivation = HGA neg)
BETA_HZ  = (13, 30)    # beta ERD (motor planning/execution; auditory listening)
THETA_HZ = (4,  8)     # theta — syllabic-rate auditory tracking (layer-2 tag only)
LOW_F_HZ = (1,  8)     # delta/theta low-frequency ERD network (layer-2 tag)

VOICE_ONSET_PCT = 67   # ASSUMPTION (single-word, RT/T_response ≈ 0.35).
                       #   Replace with subset-measured median cue→voice / cue→click ratio.


def _roles(stim, resp, motor_win, nt):
    """Build the layered role list for one grid. stim/resp/motor_win are
    (t_lo, t_hi) bin tuples (grid-dependent); nt = n_time_per_block."""
    f   = list(HGA_HZ)
    fb  = list(BETA_HZ)
    fth = list(THETA_HZ)
    flo = list(LOW_F_HZ)
    S   = list(stim)
    R   = list(resp)
    MW  = list(motor_win)

    def pct(lo, hi):
        return [max(1, round(lo / 100.0 * nt)), min(nt, round(hi / 100.0 * nt))]

    V = VOICE_ONSET_PCT

    return [

        # ════════════════ LAYER 1 — FUNCTIONAL ROLES ════════════════
        # Each role = one region/process with its full multiband signature.

        # 1. AUDITORY INPUT — responds to external speech, modality-selective,
        #    with beta ERD bundled in (HGA + beta, not two roles).
        {
            "role": "auditory", "layer": 1, "color": "#1f77b4",
            "description": "Auditory input cortex: HGA + beta-ERD to the spoken "
                           "prompt; silent to visual stimuli. Self-voice handled by "
                           "auditory_feedback.",
            "boxes": [
                {"block": "audio",   "t_bins": pct(2, 48), "f_hz": f,  "sign": "pos"},
                {"block": "audio",   "t_bins": pct(2, 48), "f_hz": fb, "sign": "neg"},  # engagement ERD
                {"block": "picture", "t_bins": S,          "f_hz": f,  "sign": "zero"},
                {"block": "reading", "t_bins": S,          "f_hz": f,  "sign": "zero"},
            ],
        },

        # 2. VISUAL INPUT — sustained HGA + beta ERD to either visual stimulus,
        #    silent to audio.
        {
            "role": "visual", "layer": 1, "color": "#2ca02c",
            "description": "Visual input cortex: sustained HGA + beta-ERD across the "
                           "on-screen image / read sentence; silent to audio.",
            "boxes": [
                {"block": "picture", "t_bins": pct(2, 48), "f_hz": f,  "sign": "pos"},
                {"block": "reading", "t_bins": pct(2, 48), "f_hz": f,  "sign": "pos"},
                {"block": "picture", "t_bins": pct(2, 48), "f_hz": fb, "sign": "neg"},
                {"block": "audio",   "t_bins": S,          "f_hz": f,  "sign": "zero"},
            ],
        },

        # 3. LEXICAL-SEMANTIC — conceptual access, CONDITION-SPECIFIC timing
        #    (picture early, audio mid, reading at the blank). Strict AND.
        {
            "role": "lexical_semantic", "layer": 1, "color": "#9467bd",
            "description": "Conceptual access at condition-specific times: picture early, "
                           "audio mid-sentence, reading at the final blank.",
            "boxes": [
                {"block": "picture", "t_bins": pct(10, 25), "f_hz": f, "sign": "pos"},
                {"block": "audio",   "t_bins": pct(30, 48), "f_hz": f, "sign": "pos"},
                {"block": "reading", "t_bins": pct(40, 48), "f_hz": f, "sign": "pos"},
            ],
        },

        # 4. HETEROMODAL CONVERGENCE — domain-general hub active in the post-sensory
        #    perception window of ALL three modalities (Forseth-style hubs).
        {
            "role": "heteromodal_convergence", "layer": 1, "color": "#8c564b",
            "description": "Multimodal hub: HGA in mid-late perception across audio AND "
                           "picture AND reading (convergent, not modality-selective).",
            "boxes": [
                {"block": "audio",   "t_bins": pct(20, 48), "f_hz": f, "sign": "pos"},
                {"block": "picture", "t_bins": pct(20, 48), "f_hz": f, "sign": "pos"},
                {"block": "reading", "t_bins": pct(20, 48), "f_hz": f, "sign": "pos"},
            ],
        },

        # 5. MAINTENANCE / WORKING MEMORY — bridges the stim->response boundary
        #    (delayed-response design). Window straddles the GO-cue (50%).
        #    NB: to STRICTLY require bridging, split each box into pre/post-cue
        #    boxes (pct(42,50) AND pct(50,58)); this lean version uses one straddle.
        {
            "role": "maintenance", "layer": 1, "color": "#7f7f7f",
            "description": "Working-memory hold: sustained HGA straddling the GO-cue, "
                           "persisting from late perception into the early response.",
            "boxes": [
                {"block": "audio",   "t_bins": pct(42, 58), "f_hz": f, "sign": "pos"},
                {"block": "picture", "t_bins": pct(42, 58), "f_hz": f, "sign": "pos"},
                {"block": "reading", "t_bins": pct(42, 58), "f_hz": f, "sign": "pos"},
            ],
        },

        # 6. PHONOLOGICAL ENCODING — cue->voice-onset, condition-shared.
        {
            "role": "phonological_encoding", "layer": 1, "color": "#e377c2",
            "description": "Pre-articulatory phonological/phonetic encoding between the "
                           "GO-cue (50%) and voice onset (~V%); condition-shared.",
            "boxes": [
                {"block": "audio",   "t_bins": pct(50, V), "f_hz": f, "sign": "pos"},
                {"block": "picture", "t_bins": pct(50, V), "f_hz": f, "sign": "pos"},
                {"block": "reading", "t_bins": pct(50, V), "f_hz": f, "sign": "pos"},
            ],
        },

        # 7. SPEECH-MOTOR — articulation HGA + beta ERD, silent during stimuli.
        {
            "role": "motor", "layer": 1, "color": "#d62728",
            "description": "Speech-motor cortex: HGA articulation at 60-85% with beta "
                           "ERD (at most zero); silent during all stimulus windows.",
            "boxes": [
                {"block": "audio",   "t_bins": MW, "f_hz": f,  "sign": "pos"},
                {"block": "picture", "t_bins": MW, "f_hz": f,  "sign": "pos"},
                {"block": "reading", "t_bins": MW, "f_hz": f,  "sign": "pos"},
                {"block": "audio",   "t_bins": MW, "f_hz": fb, "sign": "nonpos"},
                {"block": "picture", "t_bins": MW, "f_hz": fb, "sign": "nonpos"},
                {"block": "reading", "t_bins": MW, "f_hz": fb, "sign": "nonpos"},
                {"block": "audio",   "t_bins": S,  "f_hz": f,  "sign": "zero"},
                {"block": "picture", "t_bins": S,  "f_hz": f,  "sign": "zero"},
                {"block": "reading", "t_bins": S,  "f_hz": f,  "sign": "zero"},
            ],
        },

        # 8. AUDITORY FEEDBACK / SELF-MONITORING — responds to external sound but
        #    SUPPRESSED during own voice (speaker-induced suppression, Chang 2013).
        {
            "role": "auditory_feedback", "layer": 1, "color": "#17becf",
            "description": "Self-monitoring auditory cortex: HGA pos to the external "
                           "prompt, HGA suppressed during own-voice articulation (all conds).",
            "boxes": [
                {"block": "audio",   "t_bins": pct(2, 48),  "f_hz": f, "sign": "pos"},
                {"block": "audio",   "t_bins": pct(V, 85),  "f_hz": f, "sign": "neg"},
                {"block": "picture", "t_bins": pct(V, 85),  "f_hz": f, "sign": "neg"},
                {"block": "reading", "t_bins": pct(V, 85),  "f_hz": f, "sign": "neg"},
            ],
        },

        # 9. DEACTIVATION — task-negative HGA decrease (merged network_negative_1/2).
        {
            "role": "deactivation", "layer": 1, "color": "#045a8d",
            "description": "Task-negative network: HGA suppression over the early-mid "
                           "trial in all conditions (DMN-like deactivation).",
            "boxes": [
                {"block": "audio",   "t_bins": pct(10, 70), "f_hz": f, "sign": "neg"},
                {"block": "picture", "t_bins": pct(10, 70), "f_hz": f, "sign": "neg"},
                {"block": "reading", "t_bins": pct(10, 70), "f_hz": f, "sign": "neg"},
            ],
        },

        # ════════════════ LAYER 2 — SPECTRAL TAGS ═══════════════════
        # Orthogonal single-signature markers. "any" = matches on >=1 box.

        # HGA activation anywhere (functional-agnostic).
        {
            "role": "tag_hga_activation", "layer": 2, "match": "any", "color": "#aec7e8",
            "description": "HGA increase in any stim or response window (no selectivity).",
            "boxes": [
                {"block": "audio",   "t_bins": S, "f_hz": f, "sign": "pos"},
                {"block": "picture", "t_bins": S, "f_hz": f, "sign": "pos"},
                {"block": "reading", "t_bins": S, "f_hz": f, "sign": "pos"},
                {"block": "audio",   "t_bins": R, "f_hz": f, "sign": "pos"},
                {"block": "picture", "t_bins": R, "f_hz": f, "sign": "pos"},
                {"block": "reading", "t_bins": R, "f_hz": f, "sign": "pos"},
            ],
        },

        # Beta ERD engagement — the RQ1 probe (engagement WITHOUT requiring HGA).
        {
            "role": "tag_beta_erd", "layer": 2, "match": "any", "color": "#bcbd22",
            "description": "Beta (13-30) ERD during any stimulus — engagement marker; "
                           "contacts firing this with NO layer-1 HGA role are missed by "
                           "a high-gamma-only filter (RQ1).",
            "boxes": [
                {"block": "audio",   "t_bins": S, "f_hz": fb, "sign": "neg"},
                {"block": "picture", "t_bins": S, "f_hz": fb, "sign": "neg"},
                {"block": "reading", "t_bins": S, "f_hz": fb, "sign": "neg"},
            ],
        },

        # Low-frequency ERD network.
        {
            "role": "tag_low_f_erd", "layer": 2, "match": "any", "color": "#74a9cf",
            "description": "Delta/theta (1-8) ERD mid-trial — low-frequency network marker.",
            "boxes": [
                {"block": "audio",   "t_bins": pct(30, 70), "f_hz": flo, "sign": "neg"},
                {"block": "picture", "t_bins": pct(30, 70), "f_hz": flo, "sign": "neg"},
                {"block": "reading", "t_bins": pct(30, 70), "f_hz": flo, "sign": "neg"},
            ],
        },

        # Theta tracking — syllabic-rate auditory tracking (repurposes dead theta band).
        {
            "role": "tag_theta_tracking", "layer": 2, "match": "any", "color": "#fdae6b",
            "description": "Theta (4-8) increase during the spoken prompt — syllabic-rate "
                           "auditory tracking.",
            "boxes": [
                {"block": "audio", "t_bins": pct(2, 48), "f_hz": fth, "sign": "pos"},
            ],
        },

        # ════════════════ LAYER 3 — UMBRELLAS ═══════════════════════
        {
            "role": "stimulus_responsive", "layer": 3, "match": "any", "color": "#c7c7c7",
            "description": "HGA in the stimulus window of >=1 condition.",
            "boxes": [
                {"block": "audio",   "t_bins": S, "f_hz": f, "sign": "pos"},
                {"block": "picture", "t_bins": S, "f_hz": f, "sign": "pos"},
                {"block": "reading", "t_bins": S, "f_hz": f, "sign": "pos"},
            ],
        },
        {
            "role": "response_active", "layer": 3, "match": "any", "color": "#ffbb78",
            "description": "HGA in the response window of >=1 condition.",
            "boxes": [
                {"block": "audio",   "t_bins": R, "f_hz": f, "sign": "pos"},
                {"block": "picture", "t_bins": R, "f_hz": f, "sign": "pos"},
                {"block": "reading", "t_bins": R, "f_hz": f, "sign": "pos"},
            ],
        },

    ]


ERSP_POOLING_ROLES = {
    "block_order": BLOCK_ORDER,
    "ds": {
        "n_time_per_block": 30, "n_freq": 15,
        "stim_bins": [1, 15], "resp_bins": [16, 30], "pre_resp_bins": [28, 30],
        "roles": _roles(stim=(1, 15), resp=(16, 30), motor_win=(18, 25), nt=30),
    },
    "full": {
        "n_time_per_block": 300, "n_freq": 129,
        "stim_bins": [1, 150], "resp_bins": [151, 300], "pre_resp_bins": [271, 300],
        "roles": _roles(stim=(1, 150), resp=(151, 300), motor_win=(180, 255), nt=300),
    },
}