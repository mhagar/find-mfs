"""
Result classes for isotope pattern matching

Module provides dataclasses that store both aggregate scores
(for easy filtering) and detailed match information (for inspection)
"""
from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class SingleEnvelopeMatchResult:
    """
    Results from single isotope envelope matching.

    This class stores both an intensity RMSE score for ranking
    and detailed per-peak information for inspection.

    Attributes:
        num_peaks_matched: Number of peaks that matched predictions
        num_peaks_total: Total number of peaks in observed envelope
        intensity_rmse: Root-mean-square error of matched intensity
            differences (excluding base peak). Lower is better.
        match_fraction: Fraction of peaks matched (0.0 to 1.0)
            Use this for filtering
        peak_matches: Boolean array indicating which peaks matched
        predicted_envelope: The theoretical isotope envelope used for matching

    Example:
        >>> candidate: 'FormulaCandidate'
        >>> result = candidate.isotope_match_result
        >>> if result.match_fraction >= 0.8:
        ...     print(
        ...        f"Good match: "
        ...        f"{result.num_peaks_matched}/{result.num_peaks_total}"
        ...     )
        ...     print(f"Details: {result.peak_matches}")
    """
    num_peaks_matched: int
    num_peaks_total: int
    intensity_rmse: float
    match_fraction: float
    peak_matches: np.ndarray
    predicted_envelope: np.ndarray

    def __repr__(self) -> str:
        return (
            f"SingleEnvelopeMatchResult("
            f"matched={self.num_peaks_matched}/{self.num_peaks_total})"
        )


@dataclass
class MultiEnvelopeMatchResult:
    """
    Results from multi-envelope statistical matching

    This class stores both an aggregate score (mean_p_value) for easy
    filtering, and detailed per-peak p-values for inspection.

    Attributes:
        mean_p_value: Mean p-value across all peaks
            Lower values indicate better match. Use this for filtering.
        p_values: Array of p-values for each peak
        num_peaks_total: Total number of peaks
        predicted_envelope: The theoretical isotope envelope used for matching

    Example:
        >>> candidate: 'FormuleCandidate'
        >>> result = candidate.isotope_match_result
        >>> if result.mean_p_value <= 0.05:
        ...     print(f"Significant match at p={result.mean_p_value:.3f}")
        ...     print(f"Per-peak p-values: {result.p_values}")
    """
    mean_p_value: float
    p_values: np.ndarray
    num_peaks_total: int
    predicted_envelope: np.ndarray

    def __repr__(self) -> str:
        return (
            f"MultiEnvelopeMatchResult("
            f"mean_p={self.mean_p_value:.3f}, "
        )


# Type alias for either result type
IsotopeMatchResult = Union[SingleEnvelopeMatchResult, MultiEnvelopeMatchResult]
