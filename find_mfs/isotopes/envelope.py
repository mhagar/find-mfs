"""
Isotope envelope fitting using IsoSpecPy.
This module is optional, to avoid bloat for users that
don't care about isotope envelopes.

***NOTE: reminder that "monoisotopic peak" means the tallest
signal in an isotope envelope - NOT "M0".***

Requires the optional IsoSpecPy dependency
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from molmass import Formula
from molmass.elements import ELECTRON

try:
    import IsoSpecPy as iso
except ImportError:
    iso = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .results import SingleEnvelopeMatchResult
    from ..core.light_formula import LightFormula

def get_isotope_envelope(
    formula: Formula | LightFormula,
    mz_tolerance: float,
    threshold: float,
) -> np.ndarray:
    """
    Calculate the isotope envelope for a given molecular formula.

    Handles charged species by removing charge notation from formula string
    for IsoSpecPy calculation, then adjusting m/z values for electron mass,
    and charge

    Args:
        formula: Molecular formula object (may include charge)

        mz_tolerance: Minimum difference in mz values - signals less
            resolved than this will be combined

        threshold: Minimum relative intensity threshold to include
            an isotopologue into envelope

    Returns:
        Array of [m/z, intensity] pairs where intensities are scaled
        such that the monoisotopic peak is 1.0

    Raises:
        ImportError: If IsoSpecPy is not installed
    """
    if iso is None:
        raise ImportError(
            "IsoSpecPy is required for isotope envelope calculation. "
            "Install with: pip install find-mfs"
        )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"threshold argument must be between 0.0 and 1.0. "
            f"Given: {threshold}"
        )

    # Extract charge and create neutral formula string for IsoSpecPy
    charge = formula.charge
    formula_str = formula.formula

    # Remove charge notation (e.g., "C6H6+" -> "C6H6")
    if charge != 0:
        # Remove trailing + or - characters, or [ and ]
        formula_str = formula_str.rstrip('+-[]').strip('[]')

    isotope_calculator = iso.IsoThreshold(
        formula=formula_str,
        threshold=threshold,
    )

    isologues: list[tuple[float, float]] = []
    for mass, probability in isotope_calculator:
        mass: float
        probability: float

        # Adjust mass for charge state
        if charge != 0:
            # Convert neutral mass to m/z
            mz = (mass + charge * ELECTRON.mass) / abs(charge)
        else:
            mz = mass

        isologues.append(
            (mz, probability)
        )

    isologues: np.ndarray = np.array(
        isologues,
        dtype=np.float32,
    )

    isologues: np.ndarray = combine_unresolved_isotopologues(
        isologues,
        mz_tolerance=mz_tolerance,
    )

    isologues: np.ndarray = rescale_envelope(isologues)

    return isologues


def combine_unresolved_isotopologues(
        isologue_array: np.ndarray,
        mz_tolerance: float,
) -> np.ndarray:
    """
    Combines isotopologues that are within `tolerance_da` of each other.

    To combine, the intensities are summed, and the mass is changed to
    a weighted average

    Args:
        isologue_array: Array of [mass, intensity] pairs
        mz_tolerance: Mass tolerance in Daltons for combining peaks

    Returns:
        Array with combined isotopologues
    """
    sorted_idxs = np.argsort(isologue_array[:, 0])
    sorted_arr = isologue_array[sorted_idxs]

    result: list[tuple[float, float]] = []
    i = 0

    while i < len(sorted_arr):
        current_group = [sorted_arr[i]]
        current_value = sorted_arr[i, 0]

        j = i + 1

        # Find all rows with similar masses
        while (
            j < len(sorted_arr) and
            abs(sorted_arr[j, 0] - current_value) <= mz_tolerance
        ):
            current_group.append(sorted_arr[j])
            j += 1

        # Combine the group; average mass, sum intensity
        if len(current_group) > 0:
            group_array: np.ndarray = np.array(current_group)
            combined_row: tuple[float, float] = (
                np.average(  # Weighted average
                    a=group_array[:, 0],
                    weights=group_array[:, 1],
                ),
                np.sum(group_array[:, 1]),  # Intensity sum
            )
            result.append(combined_row)

        i = j

    return np.array(result)


def rescale_envelope(
    isologue_array: np.ndarray
) -> np.ndarray:
    """
    Normalizes isotope envelope intensities to monoisotopic peak
    (i.e. tallest peak)

    Args:
        isologue_array: Array of [mass, intensity] pairs

    Returns:
        Array of relative intensities
    """
    isologue_array[:, 1] = isologue_array[:, 1] / isologue_array[:, 1].max()
    return isologue_array


def _check_isospec_available():
    """
    Raise ImportError if isospec not available
    """
    if iso is None:
        raise ImportError(
            "IsoSpecPy is required for isotope envelope matching. "
            "Install with: pip install find-mfs"
        )


def match_isotope_envelope(
    formula: Formula | LightFormula,
    observed_envelope: np.ndarray,
    intsy_match_tolerance: float,
    mz_match_tolerance: float,
    simulated_envelope_mz_tolerance: float = 0.05,
    simulated_envelope_intsy_threshold: float = 0.001,
) -> 'SingleEnvelopeMatchResult':
    """
    Given a Formula and observed isotope envelope, returns detailed matching
    results including both aggregate score and per-peak information.

    Args:
        formula: Molecular formula object

        observed_envelope: Array of observed [m/z, intensity] pairs

        intsy_match_tolerance: Maximum tolerable difference between
            predicted/observed isotopologue signal relative intensity to be
            considered a match.

            This parameter should depend on the instrument's capacity for
            accurately measuring isotope ratios

        mz_match_tolerance: Maximum tolerable difference between
            predicted/observed isotopologue signal m/z value to be considered
            a match.

            This parameter should depend on the instrument's mass accuracy

        simulated_envelope_mz_tolerance: The resolution at which isotope
            envelopes will be simulated. Isotopologues less resolved than
            this value will be combined
            (i.e. intensities summed, m/z values weighted average)

        simulated_envelope_intsy_threshold: The minimum relative intensity
            to be included in a simulated isotope envelope

    Returns:
        SingleEnvelopeMatchResult containing:
        - match_fraction: Fraction of peaks matched (for filtering)
        - peak_matches: Boolean array of which peaks matched (for inspection)
        - num_peaks_matched/total: Count information
        - predicted_envelope: The theoretical envelope used
    """
    _check_isospec_available()

    if observed_envelope.ndim != 2:
        raise ValueError(
            f"Misformed `observed_envelope` array. Should be a 2D array such"
            f" that arr[:, 0] is m/z values, and arr[:, 1] is intensity values"
        )

    simulated_envelope = get_isotope_envelope(
        formula=formula,
        mz_tolerance=simulated_envelope_mz_tolerance,
        threshold=simulated_envelope_intsy_threshold,
    )

    results: np.ndarray = np.full(
        shape=observed_envelope.shape[0],
        fill_value=False,
    )

    # Iterate over each signal in observed_envelope
    for idx, signal in enumerate(observed_envelope):
        signal: np.ndarray
        diff_arr: np.ndarray = np.abs(
            simulated_envelope - signal
        )

        # check if there is a row in simulated_envelope that matches
        hits: np.ndarray = np.where(
            (diff_arr[:, 0] < mz_match_tolerance) &
            (diff_arr[:, 1] < intsy_match_tolerance)
        )[0]

        if hits.size > 0:
            # TODO: Consider whether should require a single hit
            results[idx] = True

    num_peaks_matched = int(np.sum(results))
    num_peaks_total = len(results)
    match_fraction = num_peaks_matched / num_peaks_total if num_peaks_total > 0 else 0.0

    # Import here to avoid circular dependency
    from .results import SingleEnvelopeMatchResult
    return SingleEnvelopeMatchResult(
        num_peaks_matched=num_peaks_matched,
        num_peaks_total=num_peaks_total,
        match_fraction=match_fraction,
        peak_matches=results,
        predicted_envelope=simulated_envelope
    )




