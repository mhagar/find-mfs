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
from .isotopes.config import SingleEnvelopeMatch, IsotopeMatchConfig
from .isotopes.results import SingleEnvelopeMatchResult, IsotopeMatchResult

# Utility funcs
from find_mfs.utils.filtering import (
    passes_octet_rule,
    get_rdbe,
)

# Module-level singleton for convenience function
_default_chnops_finder = None


def find_chnops(
    mass: float,
    charge: int = 0,
    error_ppm: float | None = 0.0,
    error_da: float | None = 0.0,
    adduct: str | None = None,
    **kwargs
) -> FormulaSearchResults:
    """
    Convenience function for simply finding CHNOPS formulae right away.

    Calling this function is equivalent to creating a FormulaFinder object
    then calling the FormulaFinder.find_formulae() method. This will accept
    all the same arguments as find_formulae().

    For finer control (i.e. non-CHNOPS element sets) consider instantiating
    FormulaFinder directly.

    Args:
        mass: Target mass to decompose (m/z value)

        charge: Charge state of the ion
            Default: 0 (neutral)

        error_ppm: Mass tolerance in ppm
            Either error_ppm or error_da must be specified, and the
            largest window will be used
            Default: 0.0
        error_da: Mass tolerance in Da
            Either error_ppm or error_da must be specified, and the
            largest window will be used
            Default: 0.0

        adduct: Neutral adduct formula to add/remove from the molecule.
            The adduct mass is subtracted before decomposition, then the
            adduct is added back to each candidate formula. Must be neutral
            (no '+' allowed); specify charge separately.
            Examples: "Na" for [M+Na]+, "H" for [M+H]+, "-H" for [M-H]-
            Default: None (no adduct)

        **kwargs: Additional arguments passed to FormulaFinder.find_formulae()
            (adduct, min_counts, max_counts, max_results, filter_rdbe,
            check_octet, isotope_match)

    Returns:
        FormulaSearchResults object containing candidates

    Example:
        >>> from find_mfs import find_chnops
        >>>
        >>> # Simple search
        >>> results = find_chnops(
        >>>     mass=181.071,
        >>>     error_ppm=5.0,
        >>>     charge=1,
        >>> )

        >>> # With additional filters
        >>> results = find_chnops(
        >>>     mass=203.05261,
        >>>     charge=1,
        >>>     adduct='Na',
        >>>     error_ppm=5.0,
        >>>     filter_rdbe=(0, 15),
        >>>     check_octet=True
        >>> )
    """
    global _default_chnops_finder
    if _default_chnops_finder is None:
        _default_chnops_finder = FormulaFinder('CHNOPS')

    return _default_chnops_finder.find_formulae(
        mass=mass,
        charge=charge,
        error_ppm=error_ppm,
        error_da=error_da,
        adduct=adduct,
        **kwargs
    )


__all__ = [
    # Primary API
    "FormulaFinder",
    "FormulaCandidate",
    "FormulaSearchResults",

    # Convenience function
    "find_chnops",

    # Core components
    "MassDecomposer",
    "FormulaValidator",

     # Isotope matching
    "get_isotope_envelope",
    "match_isotope_envelope",

    # Isotope matching config
    "SingleEnvelopeMatch",
    "IsotopeMatchConfig",

    # Isotope matching results
    "SingleEnvelopeMatchResult",
    "IsotopeMatchResult",

    # Utilities
    "passes_octet_rule",
    "get_rdbe",
]