"""
roi_config_concatenated.py — Functional-role conjunction templates over the
CONCATENATED [audio | picture | reading] ERSP.

Extension of roi_config.py to the *condition-structured* method: instead of a
single-condition map, each contact is represented by the three conditions
stitched along time, [audio | picture | reading]. A functional ROLE is then a
**conjunction template** — a set of boxes placed at specific (block x time x
frequency) positions, each with an expected sign. A contact is assigned a role
only if it expresses **every** box of that role's template (strict AND), where a
box "expresses" its sign via the clustering proportion gate on the score-gated
-1/0/+1 map (see lf_pool.match_roles).

Box coordinates are PER-BLOCK (1-based, inclusive), in the SAME single-condition
grid as roi_config.py:
  ds   : block = 30 time bins (stim 1-15, response 16-30); 15 freq bands.
  full : block = 300 time bins (stim 1-150, response 151-300); 129 freq rows.
The engine (lf_pool) adds the block offset (audio=0, picture=1*nt, reading=2*nt)
when slicing the concatenated map, so these stay readable.

Physiology the boxes reuse (from roi_config.py):
  HGA band  : ds f_rows [8,10] (~70-170 Hz) · full f_rows [19,44]
  stim half : ds t [1,15]  · full t [1,150]      (sensing)
  resp half : ds t [16,30] · full t [151,300]    (production + auditory feedback)

DRAFT — Lora to edit. Key caveat baked into the design: the response window is
shared by motor (articulation) and auditory feedback (hearing yourself), so by
time/frequency alone they are NOT separable — only anatomy distinguishes them.
That is why the discriminating boxes are the STIMULUS windows: auditory_input
fires on the audio stim, visual_input on the picture/reading stim. Roles overlap
on purpose (auditory's response boxes ⊃ motor's); lf_pool assigns the MOST
SPECIFIC matching role (most boxes) and also records every role a contact matches.
"""

# Block layout shared by both grids.
BLOCK_ORDER = ["audio", "picture", "reading"]


def _roles(hga, stim, resp):
    """Build the role list for one grid. hga=(f_lo,f_hi); stim/resp=(t_lo,t_hi)."""
    f = list(hga)
    S = list(stim)
    R = list(resp)
    return [
        {
            "role": "auditory",
            "description": "Auditory cortex: hears the audio STIMULUS + hears OWN VOICE "
                           "in every response (4 boxes — the audio-stim box is what makes "
                           "it auditory rather than motor).",
            "color": "#1f77b4",
            "boxes": [
                {"block": "audio",   "t_bins": S, "f_rows": f, "sign": "pos"},  # hearing the stimulus
                {"block": "audio",   "t_bins": R, "f_rows": f, "sign": "pos"},  # hearing yourself
                {"block": "picture", "t_bins": R, "f_rows": f, "sign": "pos"},  # hearing yourself
                {"block": "reading", "t_bins": R, "f_rows": f, "sign": "pos"},  # hearing yourself
            ],
        },
        {
            "role": "visual",
            "description": "Visual cortex: responds to the picture and reading STIMULI, "
                           "not to the audio stimulus (2 boxes).",
            "color": "#2ca02c",
            "boxes": [
                {"block": "picture", "t_bins": S, "f_rows": f, "sign": "pos"},
                {"block": "reading", "t_bins": S, "f_rows": f, "sign": "pos"},
            ],
        },
        {
            "role": "motor",
            "description": "Speech-motor cortex: articulation HGA in the response window of "
                           "all three conditions (3 boxes).",
            "color": "#d62728",
            "boxes": [
                {"block": "audio",   "t_bins": R, "f_rows": f, "sign": "pos"},
                {"block": "picture", "t_bins": R, "f_rows": f, "sign": "pos"},
                {"block": "reading", "t_bins": R, "f_rows": f, "sign": "pos"},
            ],
        },
        {
            "role": "multimodal_sensory",
            "description": "Amodal/heteromodal: HGA to the STIMULUS in all three conditions "
                           "regardless of modality (3 boxes).",
            "color": "#9467bd",
            "boxes": [
                {"block": "audio",   "t_bins": S, "f_rows": f, "sign": "pos"},
                {"block": "picture", "t_bins": S, "f_rows": f, "sign": "pos"},
                {"block": "reading", "t_bins": S, "f_rows": f, "sign": "pos"},
            ],
        },
    ]


ERSP_POOLING_ROLES = {
    "block_order": BLOCK_ORDER,
    "ds": {
        "n_time_per_block": 30,
        "n_freq": 15,
        "stim_bins": [1, 15],
        "resp_bins": [16, 30],
        "roles": _roles(hga=(8, 10), stim=(1, 15), resp=(16, 30)),
    },
    "full": {
        "n_time_per_block": 300,
        "n_freq": 129,
        "stim_bins": [1, 150],
        "resp_bins": [151, 300],
        "roles": _roles(hga=(19, 44), stim=(1, 150), resp=(151, 300)),
    },
}
