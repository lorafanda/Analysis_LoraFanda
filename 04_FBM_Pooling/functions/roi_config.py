"""
roi_config.py — Predefined, physiology-driven time-frequency ROIs for pooling.

The pooling in 04_FBM_Pooling is **cluster-blind and condition-blind**: the same
fixed library of time-frequency boxes is applied to every electrode x condition
ERSP, with no reliance on the 02 cluster labels or the audio/picture/reading
condition. Each ROI encodes a documented neural signature (sensory-onset HGA,
sustained processing, pre-response planning, motor execution, alpha/beta ERD,
theta retrieval, ...), so pooling turns each ERSP into a small, interpretable,
citation-anchored feature vector — the confirmatory sibling of 02's data-driven
blob / -1/0/+1 features.

Coordinates (1-BASED, INCLUSIVE on both ends):
  "ds"   — downsampled 15 freq bands x 30 time bins.
           f_rows : 1..15 index into the 15-band grid (see FREQ_BANDS in lf_features).
           t_bins : 1..30; 1-15 = stimulus, 16-30 = post-stim/response.
  "full" — 129 freq bins x 300 time bins.
           f_rows : 1..129 linear 0-500 Hz; row = round(hz/500*128) + 1.
           t_bins : 1..300; 1-150 = stimulus, 151-300 = post-stim/response.

NOTE: lf_pool.validate_roi_config() checks ds<->full Hz/time consistency and
flags mismatches (e.g. alpha_beta, broadband) — reconcile those numbers here.
"""

ERSP_POOLING_PARAMS = {

    # -- DOWNSAMPLED ERSP (15 freq bands x 30 time bins) --------------------
    "ds": [
        {
            "label": "early_sensory_onset",
            "description": "Initial cortical response to stimulus onset (auditory or visual)",
            "t_bins": [1, 5],
            "f_rows": [8, 10],
            "sign": "pos",
            "citations": [
                "Kitazawa & Asano et al. (2025) - STG HGA augmentation begins within 100ms of first phrase onset",
                "Trebuchon et al. (2020) - HGA shows favorable spatiotemporal profiles for language tasks",
            ],
        },
        {
            "label": "sustained_stimulus_processing",
            "description": "Sustained HGA during stimulus comprehension and processing",
            "t_bins": [6, 15],
            "f_rows": [8, 10],
            "sign": "pos",
            "citations": [
                "Trebuchon et al. (2020) - temporal lobe HGA sustained across full stimulus duration during language tasks",
                "Lachaux et al. (2012) - HGA reveals participation of local neural ensembles with high spatial and temporal resolution",
            ],
        },
        {
            "label": "pre_response_planning",
            "description": "Internal response formulation window prior to overt response",
            "t_bins": [16, 21],
            "f_rows": [8, 10],
            "sign": "pos",
            "citations": [
                "Kitazawa & Asano et al. (2025) - left pIFG and frontal HGA peaks 350-400ms before response onset",
                "Kitazawa & Asano et al. (2025) - left intra-hemispheric functional connectivity enhancement greatest prior to verbal response",
            ],
        },
        {
            "label": "motor_execution",
            "description": "Articulation and mouth movement during overt response period",
            "t_bins": [22, 30],
            "f_rows": [8, 10],
            "sign": "pos",
            "citations": [
                "Crone et al. (1998) - high-gamma ERS in peri-rolandic cortex somatotopically organized during motor tasks",
                "Trebuchon et al. (2020) - visual naming elicits HGA in peri-rolandic cortex and premotor regions",
            ],
        },
        {
            "label": "alpha_beta_suppression",
            "description": "ERD in alpha and beta during cortical engagement - RQ1 proof-of-concept",
            "t_bins": [3, 15],
            "f_rows": [3, 5],
            "sign": "neg",
            "citations": [
                "Crone et al. (1998) - gamma ERS and alpha/beta ERD occur simultaneously and carry distinct functional information",
                "Ray & Maunsell (2011) - beta suppression is a separate marker from broadband HGA",
                "Jia & Kohn (2011) - lower frequency power suppressed while higher frequency power increases during network activation",
            ],
        },
        {
            "label": "theta_retrieval",
            "description": "Theta increase spanning late stimulus through full response - memory and lexical retrieval",
            "t_bins": [10, 30],
            "f_rows": [2, 2],
            "sign": "pos",
            "citations": [
                "Johnson et al. (2020) - theta synchronization during encoding linked to successful later recall",
                "van Vugt et al. (2007) - theta is the primary memory-linked oscillation in iEEG; sensitive to both amplitude and duration changes",
            ],
        },
        {
            "label": "post_response_beta_rebound",
            "description": "Beta power increase after motor response - cortical return to idling state",
            "t_bins": [24, 30],
            "f_rows": [4, 5],
            "sign": "pos",
            "citations": [
                "Crone et al. (1998) - beta ERD during motor task followed by post-movement beta rebound reflecting cortical idling return",
            ],
        },
        {
            "label": "narrowband_gamma_stimulus",
            "description": "Narrowband gamma during stimulus - distinct generator from broadband HGA",
            "t_bins": [1, 15],
            "f_rows": [6, 7],
            "sign": "pos",
            "citations": [
                "Ray & Maunsell (2011) - narrowband gamma and HGA modulated in opposite directions by same stimulus; different neural generators",
                "Jia & Kohn (2011) - gamma generated by fast-spiking GABAergic interneurons; distinct from broadband HGA",
                "Smith & Kohn (2008) - fast synchrony distinct from slow correlated variability; narrowband gamma reflects fast synchrony",
            ],
        },
        {
            "label": "sustained_delta_engagement",
            "description": "Sustained low-frequency increase across full stimulus - long-range feedback circuits",
            "t_bins": [1, 15],
            "f_rows": [1, 1],
            "sign": "pos",
            "citations": [
                "Smith & Kohn (2008) - correlated variability extending across 10mm consistent with feedback projections from extrastriate cortex",
            ],
        },
        {
            "label": "early_hga_suppression",
            "description": "HGA attenuation at stimulus onset - prefrontal suppression during wh-interrogative processing",
            "t_bins": [1, 5],
            "f_rows": [8, 10],
            "sign": "neg",
            "citations": [
                "Kitazawa & Asano et al. (2025) - dorsolateral prefrontal regions show HGA attenuation at wh-interrogative onset",
                "Kitazawa & Asano et al. (2025) - right-to-left suppressive callosal flows between prefrontal regions at question onset",
            ],
        },
        {
            "label": "broadband_activation_index",
            "description": "Full-spectrum broadband shift across all bands during stimulus - power-law slope flattening index",
            "t_bins": [6, 15],
            "f_rows": [1, 10],
            "sign": "both",
            "citations": [
                "Manning et al. (2009) - broadband shifts across 2-150Hz predict single-neuron firing rates better than any narrowband feature",
                "Miller et al. (2014) - task-related broadband power shifts span all frequencies simultaneously",
                "Ray et al. (2008) - high-gamma power strongly correlated with average neuronal firing rate",
            ],
        },
    ],

    # -- FULL ERSP (129 freq bins x 300 time bins) -------------------------
    "full": [
        {
            "label": "early_sensory_onset",
            "description": "Initial cortical response to stimulus onset (auditory or visual)",
            "t_bins": [1, 50],
            "f_rows": [19, 44],
            "sign": "pos",
            "citations": [
                "Kitazawa & Asano et al. (2025) - STG HGA augmentation begins within 100ms of first phrase onset",
                "Trebuchon et al. (2020) - HGA shows favorable spatiotemporal profiles for language tasks",
            ],
        },
        {
            "label": "sustained_stimulus_processing",
            "description": "Sustained HGA during stimulus comprehension and processing",
            "t_bins": [60, 150],
            "f_rows": [19, 44],
            "sign": "pos",
            "citations": [
                "Trebuchon et al. (2020) - temporal lobe HGA sustained across full stimulus duration during language tasks",
                "Lachaux et al. (2012) - HGA reveals participation of local neural ensembles with high spatial and temporal resolution",
            ],
        },
        {
            "label": "pre_response_planning",
            "description": "Internal response formulation window prior to overt response",
            "t_bins": [160, 210],
            "f_rows": [19, 44],
            "sign": "pos",
            "citations": [
                "Kitazawa & Asano et al. (2025) - left pIFG and frontal HGA peaks 350-400ms before response onset",
                "Kitazawa & Asano et al. (2025) - left intra-hemispheric functional connectivity enhancement greatest prior to verbal response",
            ],
        },
        {
            "label": "motor_execution",
            "description": "Articulation and mouth movement during overt response period",
            "t_bins": [220, 300],
            "f_rows": [19, 44],
            "sign": "pos",
            "citations": [
                "Crone et al. (1998) - high-gamma ERS in peri-rolandic cortex somatotopically organized during motor tasks",
                "Trebuchon et al. (2020) - visual naming elicits HGA in peri-rolandic cortex and premotor regions",
            ],
        },
        {
            "label": "alpha_beta_suppression",
            "description": "ERD in alpha and beta during cortical engagement - RQ1 proof-of-concept",
            "t_bins": [30, 150],
            "f_rows": [3, 6],
            "sign": "neg",
            "citations": [
                "Crone et al. (1998) - gamma ERS and alpha/beta ERD occur simultaneously and carry distinct functional information",
                "Ray & Maunsell (2011) - beta suppression is a separate marker from broadband HGA",
                "Jia & Kohn (2011) - lower frequency power suppressed while higher frequency power increases during network activation",
            ],
        },
        {
            "label": "theta_retrieval",
            "description": "Theta increase spanning late stimulus through full response - memory and lexical retrieval",
            "t_bins": [100, 300],
            "f_rows": [2, 3],
            "sign": "pos",
            "citations": [
                "Johnson et al. (2020) - theta synchronization during encoding linked to successful later recall",
                "van Vugt et al. (2007) - theta is the primary memory-linked oscillation in iEEG",
            ],
        },
        {
            "label": "post_response_beta_rebound",
            "description": "Beta power increase after motor response - cortical return to idling state",
            "t_bins": [240, 300],
            "f_rows": [4, 6],
            "sign": "pos",
            "citations": [
                "Crone et al. (1998) - beta ERD during motor task followed by post-movement beta rebound",
            ],
        },
        {
            "label": "narrowband_gamma_stimulus",
            "description": "Narrowband gamma during stimulus - distinct generator from broadband HGA",
            "t_bins": [1, 150],
            "f_rows": [9, 19],
            "sign": "pos",
            "citations": [
                "Ray & Maunsell (2011) - narrowband gamma and HGA modulated in opposite directions by same stimulus",
                "Jia & Kohn (2011) - gamma generated by fast-spiking GABAergic interneurons; distinct from broadband HGA",
                "Smith & Kohn (2008) - fast synchrony distinct from slow correlated variability",
            ],
        },
        {
            "label": "sustained_delta_engagement",
            "description": "Sustained low-frequency increase across full stimulus - long-range feedback circuits",
            "t_bins": [1, 150],
            "f_rows": [1, 2],
            "sign": "pos",
            "citations": [
                "Smith & Kohn (2008) - correlated variability extending across 10mm consistent with feedback projections",
            ],
        },
        {
            "label": "early_hga_suppression",
            "description": "HGA attenuation at stimulus onset - prefrontal suppression during wh-interrogative processing",
            "t_bins": [1, 50],
            "f_rows": [19, 44],
            "sign": "neg",
            "citations": [
                "Kitazawa & Asano et al. (2025) - dorsolateral prefrontal regions show HGA attenuation at wh-interrogative onset",
                "Kitazawa & Asano et al. (2025) - right-to-left suppressive callosal flows between prefrontal regions at question onset",
            ],
        },
        {
            "label": "broadband_activation_index",
            "description": "Full-spectrum broadband shift across all bands during stimulus - power-law slope flattening index",
            "t_bins": [60, 150],
            "f_rows": [1, 78],
            "sign": "both",
            "citations": [
                "Manning et al. (2009) - broadband shifts across 2-150Hz predict single-neuron firing rates better than any narrowband feature",
                "Miller et al. (2014) - task-related broadband power shifts span all frequencies simultaneously",
                "Ray et al. (2008) - high-gamma power strongly correlated with average neuronal firing rate",
            ],
        },
    ],
}
