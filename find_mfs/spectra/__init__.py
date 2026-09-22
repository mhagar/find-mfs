"""
MS1 spectrum handling: data types, charge-aware signal grouping, the ion-type
vocabulary, and the mass-difference network for adduct determination.
"""

# Data types
from .envelopes import (
    SpectrumArray, to_spec_arr, ISOTOPE_SPACING,
    normalize, build_spectrum, spec_from_pairs,
)

# Envelope detection primitives
from .envelopes import (
    make_intensity_mask,
    make_shoulder_mask,
    find_isotope_envelopes,
    find_peaks_by_spacing,
)

# Charge-aware grouping (struct-of-arrays)
from .grouping import GroupedSpectrum, NoiseThreshold, group_signals

# Ion-type vocabulary (adduct model / MDN tuning knob)
from .ions import IonType, ION_VOCAB, deconv_mass

# Mass-difference network
from .network import Solution, solve_for_base

# Halogen
from .halogen import detect_halogen_envelopes, envelope_is_halogen

# File I/O
from .utils import MGFSpectrum, read_mgf

__all__ = [
    # Data types
    "SpectrumArray",
    "to_spec_arr",
    "normalize",
    "build_spectrum",
    "spec_from_pairs",
    "ISOTOPE_SPACING",
    # Envelope detection primitives
    "make_intensity_mask",
    "make_shoulder_mask",
    "find_isotope_envelopes",
    "find_peaks_by_spacing",
    # Grouping
    "GroupedSpectrum",
    "NoiseThreshold",
    "group_signals",
    # Ions
    "IonType",
    "ION_VOCAB",
    "deconv_mass",
    # Network
    "Solution",
    "solve_for_base",
    # Halogen
    "detect_halogen_envelopes",
    "envelope_is_halogen",
    # File I/O
    "MGFSpectrum",
    "read_mgf",
]
