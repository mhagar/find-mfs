"""
Mass decomposition using Bocker & Liptak algorithm,
(as implemented in SIRIUS) i.e. using an extended residue table

This algorithm was adapted from:
[Böcker & Lipták, 2007](https://link.springer.com/article/10.1007/s00453-007-0162-8)
[Böcker et. al., 2008](https://academic.oup.com/bioinformatics/article/25/2/218/218950)
"""
import numpy as np
from numba import njit
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .decomposer import Element
    from numba.typed import List as NumbaList  # nb will deprecate list reflection


@njit
def _gcd(a: int, b: int) -> int:
    """
    Calculate greatest common divisor of two integers
    """
    while b:
        a, b = b, a % b
    return a


@njit
def _is_decomposable(
        ERT: np.ndarray,
        i: int,
        m: int,
        a1: int,
) -> bool:
    """
    Check if mass m is decomposable using first i+1 elements
    """
    if m < 0:
        return False

    if ERT[int(m % a1), i] <= m:
        return True
    else:
        return False


@njit
def _decompose_integer_mass(
        ERT: np.ndarray,
        elements: list['Element'],
        mass: int,
        bounds: 'NumbaList[float]',
        max_results: int,
) -> list[list[int]]:
    """
    Find all decompositions for a specific integer mass
    """
    results = []
    num_results = 0
    k = len(elements) - 1          # Index of last element
    a1 = elements[0].integer_mass  # Mass of smallest element
    c = [0] * (k + 1)              # Current decomposition
    i = k                          # Current column in ERT
    m = mass                       # Current mass to decompose

    while (i <= k) and (num_results < max_results):
        if not _is_decomposable(ERT, i, m, a1):
            # Backtrack until we find a valid state
            while i <= k and not _is_decomposable(ERT, i, m, a1):
                m = m + c[i] * elements[i].integer_mass
                c[i] = 0
                i += 1

            # Check bounds
            while i <= k and c[i] >= bounds[i]:
                m += c[i] * elements[i].integer_mass
                c[i] = 0
                i += 1

            if i <= k:
                # Add another of current element
                m -= elements[i].integer_mass
                c[i] += 1

        else:
            # Go as deep as possible in the search tree
            while i > 0 and _is_decomposable(ERT, i - 1, m, a1):
                i -= 1

            # Found a decomposition
            if i == 0:
                c[0] = int(m // a1)
                results.append(c[:])
                num_results += 1
                i += 1  # Backtrack

            # Check bounds
            while i <= k and c[i] >= bounds[i]:
                m += c[i] * elements[i].integer_mass
                c[i] = 0
                i += 1

            if i <= k:
                # Add another of current element
                m -= elements[i].integer_mass
                c[i] += 1

    return results