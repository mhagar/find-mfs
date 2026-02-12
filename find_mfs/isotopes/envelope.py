"""
Isotope envelope fitting using IsoSpecPy.
This module is optional, to avoid bloat for users that
don't care about isotope envelopes.

***NOTE: reminder that "monoisotopic peak" means the tallest
signal in an isotope envelope - NOT "M0".***

Requires the optional IsoSpecPy dependency
"""
from typing import TYPE_CHECKING

import numpy as np
from molmass import Formula
from molmass.elements import ELECTRON

try:
    import IsoSpecPy as iso
except ImportError:
    iso = None

if TYPE_CHECKING:
    from .results import SingleEnvelopeMatchResult

def get_isotope_envelope(
    formula: Formula,
    mz_tolerance: float,
    threshold: float,
) -> np.ndarray[..., ...]:
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

    isotope_calculator: iso.IsoThreshold = iso.IsoThreshold(
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
            group_array: np.ndarray[..., ...] = np.array(current_group)
            combined_row: tuple[float, float] = (  # type: ignore
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
    isologue_array: np.ndarray[..., ...]
) -> np.ndarray[..., ...]:
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
    formula: Formula,
    observed_envelope: np.ndarray[..., ...],
    mz_match_tolerance: float,
    simulated_envelope_mz_tolerance: float = 0.1,
    simulated_envelope_intsy_threshold: float = 0.001,
) -> 'SingleEnvelopeMatchResult':
    """
    Given a Formula and observed isotope envelope, returns detailed matching
    results using RMSE scoring between aligned intensity vectors.

    For each observed peak, the closest predicted peak within
    `mz_match_tolerance` is found. Two aligned intensity vectors are built:
    observed intensities and matched predicted intensities (0.0 if no match).
    The base peak (tallest signal) is excluded from RMSE scoring since
    both envelopes are normalized to it, making it uninformative.

    Args:
        formula: Molecular formula object

        observed_envelope: Array of observed [m/z, intensity] pairs

        mz_match_tolerance: Maximum tolerable difference (in Da) between
            predicted/observed isotopologue signal m/z value to be considered
            a match.

            This parameter should depend on the instrument's mass accuracy,
            and should be set very generously.

        simulated_envelope_mz_tolerance: The resolution at which isotope
            envelopes will be simulated. Isotopologues less resolved than
            this value will be combined (i.e. intensities summed, m/z values
            weighted average). Default: 0.1

        simulated_envelope_intsy_threshold: The minimum relative intensity
            to be included in a simulated isotope envelope. Default: 0.001

    Returns:
        SingleEnvelopeMatchResult containing:
        - intensity_rmse: Root-mean-square error of matched intensity
            differences (excluding base peak). Lower is better.
        - match_fraction: Fraction of observed peaks matched to a predicted
            peak (for filtering)
        - peak_matches: Boolean array of which observed peaks matched
        - num_peaks_matched/total: Count information
        - predicted_envelope: The theoretical envelope used
    """
    _check_isospec_available()

    if observed_envelope.ndim != 2:
        raise ValueError(
            "Misformed `observed_envelope` array. Should be a 2D array such"
            " that arr[:, 0] is m/z values, and arr[:, 1] is intensity values"
        )

    simulated_envelope = get_isotope_envelope(
        formula=formula,
        mz_tolerance=simulated_envelope_mz_tolerance,
        threshold=simulated_envelope_intsy_threshold,
    )

    num_observed = observed_envelope.shape[0]
    observed_intensities = observed_envelope[:, 1].copy()
    predicted_intensities = np.zeros(num_observed, dtype=np.float64)
    peak_matches = np.full(num_observed, False)

    # For each observed peak, find the closest predicted peak within tolerance
    for idx in range(num_observed):
        obs_mz = observed_envelope[idx, 0]
        mz_diffs = np.abs(simulated_envelope[:, 0] - obs_mz)
        closest_idx = np.argmin(mz_diffs)

        if mz_diffs[closest_idx] <= mz_match_tolerance:
            predicted_intensities[idx] = simulated_envelope[closest_idx, 1]
            peak_matches[idx] = True

    # Compute RMSE of intensity differences, excluding the base peak
    # (tallest signal). Both envelopes are normalized so the base peak
    # is always 1.0, meaning it carries no discriminating information.
    base_peak_idx = np.argmax(observed_intensities)
    mask = np.ones(num_observed, dtype=bool)
    mask[base_peak_idx] = False
    obs_for_rmse = observed_intensities[mask]
    pred_for_rmse = predicted_intensities[mask]

    if len(obs_for_rmse) > 0:
        intensity_rmse = float(np.sqrt(
            np.mean((obs_for_rmse - pred_for_rmse) ** 2)
        ))
    else:
        intensity_rmse = 0.0

    num_peaks_matched = int(np.sum(peak_matches))
    num_peaks_total = num_observed
    match_fraction = num_peaks_matched / num_peaks_total \
        if num_peaks_total > 0 else 0.0

    # Import here to avoid circular dependency
    from .results import SingleEnvelopeMatchResult
    return SingleEnvelopeMatchResult(
        num_peaks_matched=num_peaks_matched,
        num_peaks_total=num_peaks_total,
        intensity_rmse=intensity_rmse,
        match_fraction=match_fraction,
        peak_matches=peak_matches,
        predicted_envelope=simulated_envelope,
    )


def match_isotope_envelope_series(
    formula: Formula,
    observed_envelopes: list[np.ndarray[float, float]],
    mz_match_tolerance: float,
    simulated_envelope_mz_tolerance: float = 0.05,
    simulated_envelope_intsy_threshold: float = 0.001,
):
    """
    *** PLACEHOLDER ***
    This function will be implemented in the future;
    (statistical approach to isotope envelope matching)
    """
    raise NotImplemented(
        "This function has not yet been implemented"
    )



