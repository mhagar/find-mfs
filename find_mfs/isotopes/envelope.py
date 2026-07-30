"""
Isotope envelope fitting using IsoSpecPy.

***NOTE: reminder that "monoisotopic peak" means the tallest
signal in an isotope envelope - NOT "M0".***
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import IsoSpecPy as iso
from molmass import Formula
from molmass.elements import ELECTRON

if TYPE_CHECKING:
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
    """
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

