"""
lf_micromacro.py
================

Helper functions unique to the MicroEPI micro/macro ERSP comparison pipeline.

Used by:
    01_FBM_Analysis/scripts/11_MicroMacroComparison_Analysis.ipynb

Scope:
    Only functions that are specific to MicroEPI micro/macro comparison live here.
    Shared logic (photodiode event extraction, trial TSV loading/validation, ERSP
    computation, WM referencing) is reused from existing helpers:
        - lf_io_utils.py
        - lf_trials.py
        - lf_ersp.py
        - 01_PD_extract_* (photodiode square-wave extraction logic)
"""

# -----------------------------------------------------------------------------
# .mat loader
# -----------------------------------------------------------------------------
def load_microepi_mat(path):
    """
    Load a MicroEPI export .mat file.

    Expected fields:
        - dataEcog        : (n_macro_channels, n_samples) single   macro channels
        - dataMicroDown   : (n_micro_channels, n_samples) double   micro channels (downsampled to fs_ecog)
        - chansEcog       : struct array of macro channel metadata
        - chansMicro      : struct array of micro channel metadata
        - photodiode      : (1, n_samples)                         square-wave event trace
        - fileOriginalInfo, diagnostics, evSample, evValue (ignored / empty)

    Sampling rate:
        fs = 2048 Hz (dataEcog rate; dataMicroDown already matches)

    Returns
    -------
    dict with keys:
        'data_ecog', 'data_micro', 'chans_ecog', 'chans_micro', 'photodiode', 'fs'
    """
    pass


# -----------------------------------------------------------------------------
# Channel pairing: micro shaft -> first macro contact
# -----------------------------------------------------------------------------
def pair_micro_to_macro(chans_micro, chans_ecog):
    """
    Pair micro channels to their corresponding first-contact macro channel.

    Rule:
        micro name = '<prefix>m<N>'  ->  matched macro = '<prefix>1'

        e.g. AGm1, AGm2, ..., AGm12  ->  AG1

    Returns
    -------
    dict
        { shaft_prefix: { 'macro': '<prefix>1',
                          'micros': ['<prefix>m1', '<prefix>m2', ...] } }
    """
    pass


# -----------------------------------------------------------------------------
# Tetrode grouping
# -----------------------------------------------------------------------------
def group_into_tetrodes(micro_list):
    """
    Group an ordered list of micro channel names into tetrodes of 4.

    Examples:
        8 micros  -> 2 tetrodes -> 2 groups of 4
        12 micros -> 3 tetrodes -> 3 groups of 4

    Returns
    -------
    list of lists, each inner list has length 4
    """
    pass


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_macro_ersp(*args, **kwargs):
    """
    Plot a single macro channel ERSP (one figure per shaft).
    """
    pass


def plot_tetrode_ersp_2x2(*args, **kwargs):
    """
    Plot a 2x2 subplot of ERSPs for the 4 micro channels of one tetrode.
    One figure per tetrode.
    """
    pass
