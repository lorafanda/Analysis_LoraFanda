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
ULTRA_HFA_HZ = (200, 500)   # ultra high-frequency activity (>200 Hz; HFOs / very-high-gamma)

VOICE_ONSET_PCT = 67   # ASSUMPTION (single-word, RT/T_response ≈ 0.35).
                       #   Replace with subset-measured median cue→voice / cue→click ratio.


def _roles(stim, resp, motor_win, nt):
    """Build the layered role list for one grid. stim/resp/motor_win are
    (t_lo, t_hi) bin tuples (grid-dependent); nt = n_time_per_block."""
    f   = list(HGA_HZ)
    fb  = list(BETA_HZ)
    fth = list(THETA_HZ)
    flo = list(LOW_F_HZ)
    fu  = list(ULTRA_HFA_HZ)
    S   = list(stim)
    R   = list(resp)
    MW  = list(motor_win)

    def pct(lo, hi):
        return [max(1, round(lo / 100.0 * nt)), min(nt, round(hi / 100.0 * nt))]

    V = VOICE_ONSET_PCT

    return [

        # ════════════════ LAYER 1 — FUNCTIONAL ROLES ════════════════
        # Each role = one region/process with its full multiband signature.

        # 1. AUDITORY INPUT — early HGA to spoken prompt; silent during visual stim.
        {
            "role": "auditory", "layer": 1, "color": "#1f77b4", "thr": 1.5, "frac": 0.10,
            "description": "Auditory input cortex: early HGA to the spoken prompt "
                           "(7-14%); silent (zero) to the visual picture stimulus.",
            "boxes": [
                {"block": "audio",   "t_bins": pct(7, 14),  "f_hz": [79, 151],  "sign": "pos"},
                {"block": "picture", "t_bins": pct(3, 24),  "f_hz": [52, 185],  "sign": "zero"},
            ],
        },

        # 1b. AUDITORY v2 — designer-fitted: broadband HGA onset + early low-freq decrease.
        {
            "role": "auditory_v2", "layer": 1, "color": "#539fd4", "thr": 1.5, "frac": 0.25,
            "description": "Auditory cortex (lenient): HGA onset 8-28% (39-102 Hz) and "
                           "concurrent low-frequency decrease 12-34% (6-26 Hz) to the spoken prompt.",
            "boxes": [
                {"block": "audio", "t_pct": [8,  28], "f_hz": [39, 102], "sign": "pos"},
                {"block": "audio", "t_pct": [12, 34], "f_hz": [6,  26],  "sign": "neg"},
            ],
        },

        # 2. VISUAL INPUT — split by modality (HGA to the image vs to the sentence),
        #    each silent to audio. (beta box dropped; picture & reading no longer conjoined.)
        {
            "role": "visual_picture", "layer": 1, "color": "#2ca02c", "thr": 2.0, "frac": 0.30,
            "description": "Visual cortex (picture): HGA across the image-viewing "
                           "window; silent to the audio prompt.",
            "boxes": [
                {"block": "picture", "t_bins": pct(2, 48), "f_hz": f, "sign": "pos"},
                {"block": "audio",   "t_bins": S,          "f_hz": f, "sign": "zero"},
            ],
        },
        {
            "role": "visual_reading", "layer": 1, "color": "#1f9e6e", "thr": 2.0, "frac": 0.30,
            "description": "Visual cortex (reading): HGA across the sentence-reading "
                           "window; silent to the audio prompt.",
            "boxes": [
                {"block": "reading", "t_bins": pct(2, 48), "f_hz": f, "sign": "pos"},
                {"block": "audio",   "t_bins": S,          "f_hz": f, "sign": "zero"},
            ],
        },

        # 2b. PICTURE STIM — low-freq pos onset + beta ERD mid-picture.
        {
            "role": "picture_stim", "layer": 1, "color": "#d4a017", "thr": 2.0, "frac": 0.25,
            "description": "Picture stimulus response: low-frequency increase at onset "
                           "(6-10%) followed by beta ERD during the viewing window (30-46%).",
            "boxes": [
                {"block": "picture", "t_bins": pct(6, 10),  "f_hz": [5, 34],   "sign": "pos"},
                {"block": "picture", "t_bins": pct(30, 46), "f_hz": [13, 24],  "sign": "neg"},
            ],
        },

        # 2c. PIC STIM — designer-fitted picture onset (t_pct); slightly wider beta band.
        {
            "role": "pic_stim", "layer": 1, "color": "#e9c46a", "thr": 2.0, "frac": 0.25,
            "description": "Picture-specific: early low-freq onset at 6-10% (wide band 5-34 Hz) "
                           "and mid-stimulus beta decrease at 30-46% (13-26 Hz).",
            "boxes": [
                {"block": "picture", "t_pct": [6,  10], "f_hz": [5,  34], "sign": "pos"},
                {"block": "picture", "t_pct": [30, 46], "f_hz": [13, 26], "sign": "neg"},
            ],
        },

        # 3. LEXICAL-SEMANTIC — conceptual access, CONDITION-SPECIFIC timing
        #    (picture early, audio mid, reading at the blank). Strict AND.
        {
            "role": "lexical_semantic", "layer": 1, "color": "#9467bd", "thr": 2.0, "frac": 0.30,
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
            "role": "heteromodal_convergence", "layer": 1, "color": "#8c564b", "thr": 2.0, "frac": 0.30,
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
            "role": "maintenance", "layer": 1, "color": "#7f7f7f", "thr": 2.0, "frac": 0.30,
            "description": "Working-memory hold: sustained HGA straddling the GO-cue, "
                           "persisting from late perception into the early response.",
            "boxes": [
                {"block": "audio",   "t_bins": pct(42, 58), "f_hz": f, "sign": "pos"},
                {"block": "picture", "t_bins": pct(42, 58), "f_hz": f, "sign": "pos"},
                {"block": "reading", "t_bins": pct(42, 58), "f_hz": f, "sign": "pos"},
            ],
        },

        # 5b. PREMOTOR PLANNING — pre-response HGA in all 3 conditions + nonpos HFO.
        {
            "role": "premotor planning", "layer": 1, "color": "#ff7f0e", "thr": 0.5, "frac": 0.25,
            "description": "Premotor planning: pre-response HGA broadband increase "
                           "across all three conditions with suppressed high-frequency oscillations.",
            "boxes": [
                {"block": "picture", "t_pct": [10, 31], "f_hz": [72, 150],  "sign": "pos"},
                {"block": "audio",   "t_pct": [38, 56], "f_hz": [71, 151],  "sign": "pos"},
                {"block": "reading", "t_pct": [33, 48], "f_hz": [64, 150],  "sign": "pos"},
                {"block": "audio",   "t_pct": [6,  94], "f_hz": [261, 495], "sign": "nonpos"},
                {"block": "picture", "t_pct": [3,  99], "f_hz": [259, 493], "sign": "nonpos"},
                {"block": "reading", "t_pct": [4,  99], "f_hz": [241, 493], "sign": "nonpos"},
            ],
        },

        # 6. PHONOLOGICAL ENCODING — cue->voice-onset, condition-shared.
        {
            "role": "phonological_encoding", "layer": 1, "color": "#e377c2", "thr": 2.0, "frac": 0.30,
            "description": "Pre-articulatory phonological/phonetic encoding between the "
                           "GO-cue (50%) and voice onset (~V%); condition-shared.",
            "boxes": [
                {"block": "audio",   "t_bins": pct(50, V), "f_hz": f, "sign": "pos"},
                {"block": "picture", "t_bins": pct(50, V), "f_hz": f, "sign": "pos"},
                {"block": "reading", "t_bins": pct(50, V), "f_hz": f, "sign": "pos"},
            ],
        },

        # 7. SPEECH-MOTOR — articulation HGA + beta ERD (nonpos), silent during stimuli.
        {
            "role": "motor", "layer": 1, "color": "#d62728", "thr": 2.0, "frac": 0.30,
            "description": "Speech-motor cortex: HGA at 60-85% in all conditions, beta "
                           "not-activating (nonpos) in same window; silent (zero) during "
                           "stimulus window (0-50%).",
            "boxes": [
                {"block": "audio",   "t_pct": [60, 85], "f_hz": [70, 150], "sign": "pos"},
                {"block": "picture", "t_pct": [60, 85], "f_hz": [70, 150], "sign": "pos"},
                {"block": "reading", "t_pct": [60, 85], "f_hz": [70, 150], "sign": "pos"},
                {"block": "audio",   "t_pct": [60, 85], "f_hz": [13,  30], "sign": "nonpos"},
                {"block": "picture", "t_pct": [60, 85], "f_hz": [13,  30], "sign": "nonpos"},
                {"block": "reading", "t_pct": [60, 85], "f_hz": [13,  30], "sign": "nonpos"},
                {"block": "audio",   "t_pct": [0,  50], "f_hz": [70, 150], "sign": "zero"},
                {"block": "picture", "t_pct": [0,  50], "f_hz": [70, 150], "sign": "zero"},
                {"block": "reading", "t_pct": [0,  50], "f_hz": [70, 150], "sign": "zero"},
            ],
        },

        # 7b. SPEECH-MOTOR v2 — lenient per-condition HGA + beta ERD, stim silent.
        {
            "role": "motor_v2", "layer": 1, "color": "#e57373", "thr": 1.0, "frac": 0.25,
            "description": "Speech-motor (lenient): per-condition HGA at 62-85% (lower thr), "
                           "silent (zero HGA all freqs) during stimulus, beta decrease mid-trial.",
            "boxes": [
                {"block": "audio",   "t_pct": [63, 85], "f_hz": [68, 150], "sign": "pos"},
                {"block": "picture", "t_pct": [62, 81], "f_hz": [68, 145], "sign": "pos"},
                {"block": "reading", "t_pct": [62, 82], "f_hz": [77, 151], "sign": "pos"},
                {"block": "audio",   "t_pct": [0,  51], "f_hz": [68, 500], "sign": "zero"},
                {"block": "picture", "t_pct": [2,  54], "f_hz": [66, 500], "sign": "zero"},
                {"block": "reading", "t_pct": [1,  55], "f_hz": [70, 500], "sign": "zero"},
                {"block": "reading", "t_pct": [33, 56], "f_hz": [0,  31],  "sign": "neg"},
                {"block": "picture", "t_pct": [35, 57], "f_hz": [0,  37],  "sign": "neg"},
                {"block": "audio",   "t_pct": [34, 57], "f_hz": [0,  31],  "sign": "neg"},
            ],
        },

        # 8. AUDITORY FEEDBACK / SELF-MONITORING — responds to external sound but
        #    SUPPRESSED during own voice (speaker-induced suppression, Chang 2013).
        {
            "role": "auditory_feedback", "layer": 1, "color": "#17becf", "thr": 2.0, "frac": 0.30,
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
            "role": "deactivation", "layer": 1, "color": "#045a8d", "thr": 2.0, "frac": 0.30,
            "description": "Task-negative network: HGA suppression over the early-mid "
                           "trial in all conditions (DMN-like deactivation).",
            "boxes": [
                {"block": "audio",   "t_bins": pct(10, 70), "f_hz": f, "sign": "neg"},
                {"block": "picture", "t_bins": pct(10, 70), "f_hz": f, "sign": "neg"},
                {"block": "reading", "t_bins": pct(10, 70), "f_hz": f, "sign": "neg"},
            ],
        },

        # 9b. NN — broadband HGA suppression across all conditions (designer-fitted).
        {
            "role": "NN", "layer": 1, "color": "#6c5b7b", "thr": 1.5, "frac": 0.30,
            "description": "Neural negativity: sustained broadband HGA suppression "
                           "across audio (9-37%), picture (13-54%), and reading (11-54%). "
                           "Broad-spectrum task-negative network.",
            "boxes": [
                {"block": "audio",   "t_pct": [9,  37], "f_hz": [122, 381], "sign": "neg"},
                {"block": "picture", "t_pct": [13, 54], "f_hz": [147, 403], "sign": "neg"},
                {"block": "reading", "t_pct": [11, 54], "f_hz": [112, 401], "sign": "neg"},
            ],
        },

        # ════════════════ LAYER 2 — SPECTRAL TAGS ═══════════════════
        # Orthogonal single-signature markers. "any" = matches on >=1 box.

        # HGA activation anywhere (functional-agnostic).
        {
            "role": "tag_hga_activation", "layer": 2, "match": "any", "color": "#aec7e8", "thr": 2.0, "frac": 0.20,
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
            "role": "tag_beta_erd", "layer": 2, "match": "any", "color": "#bcbd22", "thr": 2.0, "frac": 0.20,
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
            "role": "tag_low_f_erd", "layer": 2, "match": "any", "color": "#74a9cf", "thr": 2.0, "frac": 0.20,
            "description": "Delta/theta (1-8) ERD mid-trial — low-frequency network marker.",
            "boxes": [
                {"block": "audio",   "t_bins": pct(30, 70), "f_hz": flo, "sign": "neg"},
                {"block": "picture", "t_bins": pct(30, 70), "f_hz": flo, "sign": "neg"},
                {"block": "reading", "t_bins": pct(30, 70), "f_hz": flo, "sign": "neg"},
            ],
        },

        # Theta tracking — syllabic-rate auditory tracking (repurposes dead theta band).
        {
            "role": "tag_theta_tracking", "layer": 2, "match": "any", "color": "#fdae6b", "thr": 2.0, "frac": 0.20,
            "description": "Theta (4-8) increase during the spoken prompt — syllabic-rate "
                           "auditory tracking.",
            "boxes": [
                {"block": "audio", "t_bins": pct(2, 48), "f_hz": fth, "sign": "pos"},
            ],
        },

        # Ultra-HFA — very-high-frequency (>200 Hz) burst in the MIDDLE 40% of any
        # stimulus OR response window (HFOs / very-high-gamma).
        {
            "role": "ultra_hfa", "layer": 2, "match": "any", "color": "#ff1493", "thr": 2.0, "frac": 0.20,
            "description": "Ultra high-frequency activity (>200 Hz) in the middle 40% of any "
                           "stimulus or response window (HFOs / very-high-gamma). NB: 200-400 Hz "
                           "overlaps mains harmonics + speech EMG — interpret with care.",
            "boxes": [
                {"block": "audio",   "t_bins": pct(15, 35), "f_hz": fu, "sign": "pos"},
                {"block": "picture", "t_bins": pct(15, 35), "f_hz": fu, "sign": "pos"},
                {"block": "reading", "t_bins": pct(15, 35), "f_hz": fu, "sign": "pos"},
                {"block": "audio",   "t_bins": pct(65, 85), "f_hz": fu, "sign": "pos"},
                {"block": "picture", "t_bins": pct(65, 85), "f_hz": fu, "sign": "pos"},
                {"block": "reading", "t_bins": pct(65, 85), "f_hz": fu, "sign": "pos"},
            ],
        },

        # ════════════════ LAYER 3 — UMBRELLAS ═══════════════════════
        {
            "role": "stimulus_active", "layer": 3, "match": "any", "color": "#c7c7c7", "thr": 2.0, "frac": 0.20,
            "description": "HGA in the stimulus window of >=1 condition.",
            "boxes": [
                {"block": "audio",   "t_bins": S, "f_hz": f, "sign": "pos"},
                {"block": "picture", "t_bins": S, "f_hz": f, "sign": "pos"},
                {"block": "reading", "t_bins": S, "f_hz": f, "sign": "pos"},
            ],
        },
        {
            "role": "response_active", "layer": 3, "match": "any", "color": "#ffbb78", "thr": 2.0, "frac": 0.20,
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