"""
find_mfs: A Python package for finding molecular formulae from mass spectrometry data.

This package implements an efficient mass decomposition algorithm based on
Bocker & Liptak's "A Fast and Simple Algorithm for the Money Changing Problem"
with additional chemical validation rules and optional isotope envelope matching.
"""

__version__ = "0.1.0"
__author__ = "Mostafa Hagar"

# Main API
from .core.finder import FormulaFinder, FormulaCandidate
from .core.results import FormulaSearchResults

# Lower-level components
from .core.decomposer import MassDecomposer
from .core.validator import FormulaValidator

# Isotope matching functions
from .isotopes.envelope import (
    get_isotope_envelope,
    match_isotope_envelope,
)

# Isotope matching configs and results objects
from .isotopes.config import SingleEnvelopeMatch, MultiEnvelopeMatch, IsotopeMatchConfig
from .isotopes.results import SingleEnvelopeMatchResult, IsotopeMatchResult

# Utility funcs
from find_mfs.utils.filtering import (
    passes_octet_rule,
    get_rdbe,
)

__all__ = [
    # Primary API
    "FormulaFinder",
    "FormulaCandidate",
    "FormulaSearchResults",

    # Core components
    "MassDecomposer",
    "FormulaValidator",

     # Isotope matching
    "get_isotope_envelope",
    "match_isotope_envelope",
    # "match_isotope_envelope_series", WORK IN PROGRESS

    # Isotope matching config
    "SingleEnvelopeMatch",
    # "MultiEnvelopeMatch",            WORK IN PROGRESS
    "IsotopeMatchConfig",

    # Isotope matching results
    "SingleEnvelopeMatchResult",
    # "MultiEnvelopeMatchResult",      WORK IN PROGRESS
    "IsotopeMatchResult",

    # Utilities
    "passes_octet_rule",
    "get_rdbe",
]