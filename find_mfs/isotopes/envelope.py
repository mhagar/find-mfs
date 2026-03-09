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

    # Build a neutral formula string for IsoSpecPy without relying on parsing
    # charge-annotated formula strings (e.g. "[C6H11O6]2+").
    charge = formula.charge
    pairs = [
        (sym, item.count)
        for sym, item in formula.composition().items()
        if sym and sym != 'e-' and item.count > 0
    ]
    if not pairs:
        return np.empty((0, 2), dtype=np.float32)

    pairs = sorted(
        pairs,
        key=lambda p: (0,) if p[0] == 'C' else (1,) if p[0] == 'H' else (2, p[0]),
    )
    formula_str = ''.join(
        sym if count == 1 else f'{sym}{count}'
        for sym, count in pairs
    )

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
            mz = (mass - charge * ELECTRON.mass) / abs(charge)
        else:
            mz = mass

        isologues.append(
            (mz, probability)
        )

    # Note: float32 is used here for the old match_isotope_envelope() path.
    # The fast path uses float64 throughout via IsoSpecPy's C++ doubles.
    # This causes small RMSE differences (<0.01 absolute) between paths,
    # which is acceptable.
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
    formula: Formula | LightFormula,
    observed_envelope: np.ndarray,
    mz_match_tolerance: float,
    simulated_envelope_mz_tolerance: float = 0.05,
    simulated_envelope_intsy_threshold: float = 0.001,
) -> 'SingleEnvelopeMatchResult':
    """
    Given a Formula and observed isotope envelope, returns detailed matching
    results using RMSE scoring between aligned intensity vectors.

    Uses Cython + C++ IsoSpecPy for fast scoring. For each observed peak,
    the closest predicted peak within ``mz_match_tolerance`` is found.
    The base peak is excluded from RMSE scoring since both envelopes are
    normalized to it.

    Args:
        formula: Molecular formula object

        observed_envelope: Array of observed [m/z, intensity] pairs

        mz_match_tolerance: Maximum tolerable difference (in Da) between
            predicted/observed isotopologue signal m/z value to be considered
            a match.

        simulated_envelope_mz_tolerance: The resolution at which isotope
            envelopes will be simulated. Default: 0.05

        simulated_envelope_intsy_threshold: The minimum relative intensity
            to be included in a simulated isotope envelope. Default: 0.001

    Returns:
        SingleEnvelopeMatchResult containing RMSE, match fraction, etc.
    """
    _check_isospec_available()

    if observed_envelope.ndim != 2:
        raise ValueError(
            "Misformed `observed_envelope` array. Should be a 2D array such"
            " that arr[:, 0] is m/z values, and arr[:, 1] is intensity values"
        )

    # Extract symbols, counts, and charge from the formula object
    from .results import SingleEnvelopeMatchResult
    from ..core.light_formula import LightFormula as _LF

    if isinstance(formula, _LF):
        pairs = list(formula._iter_nonzero_items())
    else:
        pairs = [
            (sym, item.count)
            for sym, item in formula.composition().items()
            if sym and sym != 'e-' and item.count > 0
        ]

    if not pairs:
        n_obs = observed_envelope.shape[0]
        return SingleEnvelopeMatchResult(
            num_peaks_matched=0,
            num_peaks_total=n_obs,
            intensity_rmse=1.0,
            match_fraction=0.0,
            peak_matches=np.full(n_obs, False),
            predicted_envelope=np.empty((0, 2), dtype=np.float64),
        )

    symbols = [p[0] for p in pairs]
    counts = [p[1] for p in pairs]
    charge = formula.charge

    predicted_envelope = get_isotope_envelope(
        formula=formula,
        mz_tolerance=simulated_envelope_mz_tolerance,
        threshold=simulated_envelope_intsy_threshold,
    )

    from ._isospec import score_isotope_batch as _cython_batch
    counts_2d = np.array([counts], dtype=np.int32)
    rmse_arr, mf_arr, nm_arr, pm_arr = _cython_batch(
        symbols, counts_2d, charge, observed_envelope,
        mz_match_tolerance, simulated_envelope_mz_tolerance,
        simulated_envelope_intsy_threshold,
    )

    return SingleEnvelopeMatchResult(
        num_peaks_matched=int(nm_arr[0]),
        num_peaks_total=observed_envelope.shape[0],
        intensity_rmse=float(rmse_arr[0]),
        match_fraction=float(mf_arr[0]),
        peak_matches=pm_arr[0].astype(bool),
        predicted_envelope=predicted_envelope,
    )


def score_isotope_batch(
    symbols: list[str] | tuple[str, ...],
    counts_2d: np.ndarray,
    charge: int,
    observed_envelope: np.ndarray,
    mz_match_tolerance: float,
    simulated_mz_tolerance: float = 0.05,
    simulated_intensity_threshold: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch isotope envelope scoring for multiple candidates at once.

    Uses Cython + C++ for high performance.

    Args:
        symbols: Element symbols (e.g., ['C', 'H', 'N', 'O', 'P', 'S'])
        counts_2d: int32 array of shape (N, n_elements) with atom counts
        charge: Ion charge state
        observed_envelope: 2D array of [m/z, intensity] pairs (normalized)
        mz_match_tolerance: Max m/z difference for peak matching (Da)
        simulated_mz_tolerance: Resolution for combining isotopologues
        simulated_intensity_threshold: Min relative intensity threshold

    Returns:
        Tuple of (rmse_arr, match_frac_arr, n_matched_arr, peak_matches_2d)
        where peak_matches_2d is int8 array of shape (N, n_obs) with per-peak
        match booleans (1=matched, 0=unmatched).
    """
    from ._isospec import score_isotope_batch as _cython_batch
    return _cython_batch(
        symbols, counts_2d, charge, observed_envelope,
        mz_match_tolerance, simulated_mz_tolerance,
        simulated_intensity_threshold,
    )
