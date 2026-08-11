"""
MS1 spectrum parsing: isotope envelope detection, adduct grouping, and formula query
"""

# Pipeline (main API)
from .pipeline import AnnotatedSpectrum, parse_spectrum, query_envelopes, query_spectrum

# Data types
from .envelopes import (
    SpectrumArray, to_spec_arr, ISOTOPE_SPACING,
    normalize, build_spectrum, spec_from_pairs,
)

# Envelope detection
from .envelopes import (
    make_intensity_mask,
    make_shoulder_mask,
    find_isotope_envelopes,
    find_peaks_by_spacing,
)

# Adducts
from .adducts import (
    build_adduct_pair_deltas,
    find_adduct_groups,
    identify_adduct,
    POSITIVE_ADDUCT_SPECS,
    NEGATIVE_ADDUCT_SPECS,
    POSITIVE_PAIR_DELTAS,
    NEGATIVE_PAIR_DELTAS,
)

# Halogen
from .halogen import detect_halogen_envelopes

# File I/O
from .utils import MGFSpectrum, read_mgf

__all__ = [
    # Pipeline
    "AnnotatedSpectrum",
    "parse_spectrum",
    "query_envelopes",
    "query_spectrum",
    # Data types
    "SpectrumArray",
    "to_spec_arr",
    "normalize",
    "build_spectrum",
    "spec_from_pairs",
    "ISOTOPE_SPACING",
    # Envelope detection
    "make_intensity_mask",
    "make_shoulder_mask",
    "find_isotope_envelopes",
    "find_peaks_by_spacing",
    # Adducts
    "build_adduct_pair_deltas",
    "find_adduct_groups",
    "identify_adduct",
    "POSITIVE_ADDUCT_SPECS",
    "NEGATIVE_ADDUCT_SPECS",
    "POSITIVE_PAIR_DELTAS",
    "NEGATIVE_PAIR_DELTAS",
    # Halogen
    "detect_halogen_envelopes",
    # File I/O
    "MGFSpectrum",
    "read_mgf",
]
