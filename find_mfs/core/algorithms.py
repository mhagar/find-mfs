"""
Mass decomposition using Bocker & Liptak algorithm,
(as implemented in SIRIUS) i.e. using an extended residue table

This algorithm was adapted from:
[Böcker & Lipták, 2007](https://link.springer.com/article/10.1007/s00453-007-0162-8)
[Böcker et. al., 2008](https://academic.oup.com/bioinformatics/article/25/2/218/218950)
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _gcd(a: int, b: int) -> int:
    """
    Calculate greatest common divisor of two integers
    """
    while b:
        a, b = b, a % b
    return a


@njit(cache=True, inline='always')
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


@njit(cache=True, fastmath=True)
def _decompose_mass_range(
        ERT: np.ndarray,
        integer_masses: np.ndarray,
        real_masses: np.ndarray,
        bounds: np.ndarray,
        min_values: np.ndarray,
        min_int: int,
        max_int: int,
        original_min_mass: float,
        original_max_mass: float,
        charge_mass_offset: float,
        max_results: int,
        rdbe_coeffs: np.ndarray,
        rdbe_min: float,
        rdbe_max: float,
        check_octet: bool,
        charge_parity_even: bool,
        do_rdbe_filter: bool,
) -> np.ndarray:
    """
    Consolidated decomposition across an integer mass range.

    Iterates over all integer masses in [min_int, max_int], performs
    decomposition with bounds checking, applies min_values and exact
    mass filtering internally, and returns valid element count vectors.

    Uses fastmath=True to enable FMA fusion in the exact mass dot product
    (vfmadd231sd instead of separate vmulsd+vaddsd on the critical path).

    Pre-allocates the output buffer at max_results to avoid dynamic growth
    and the NRT_incref/NRT_decref calls it would cause per loop iteration.

    When do_rdbe_filter is True, applies RDBE range and octet rule checks
    at the leaf level, avoiding writing filtered-out candidates to the
    result array entirely.

    Returns:
        2D int32 array of shape (N, num_elements) with valid counts
    """
    num_elements = len(integer_masses)
    a1 = integer_masses[0]
    k = num_elements - 1

    # Pre-allocate output buffer at max_results to avoid dynamic growth.
    # This eliminates the result array reassignment that causes LLVM to
    # emit NRT_incref/NRT_decref on every outer loop iteration.
    result = np.empty((max_results, num_elements), dtype=np.int32)
    count = 0
    # Track total mass-valid candidates (before RDBE filter) to preserve
    # the same stopping semantics as master: stop after max_results
    # candidates pass the mass filter, regardless of RDBE outcome.
    mass_valid_count = 0

    c = np.zeros(num_elements, dtype=np.int64)

    for m_target in range(min_int, max_int + 1):
        # Reset state for this integer mass
        for j in range(num_elements):
            c[j] = 0
        i = k
        m = np.int64(m_target)

        while i <= k and mass_valid_count < max_results:
            if not _is_decomposable(ERT, i, m, a1):
                # Backtrack until decomposable
                while i <= k and not _is_decomposable(ERT, i, m, a1):
                    m += c[i] * integer_masses[i]
                    c[i] = 0
                    i += 1
                # Check bounds
                while i <= k and c[i] >= bounds[i]:
                    m += c[i] * integer_masses[i]
                    c[i] = 0
                    i += 1
                if i <= k:
                    m -= integer_masses[i]
                    c[i] += 1
            else:
                # Descend as deep as possible
                while i > 0 and _is_decomposable(ERT, i - 1, m, a1):
                    i -= 1

                if i == 0:
                    c[0] = m // a1

                    # Check bounds for element 0
                    if c[0] <= bounds[0]:
                        # Compute exact mass and RDBE with min_values applied
                        # in a single fused loop. RDBE is always computed
                        # (branch-free inner loop); when not filtering,
                        # zero coeffs and -inf/+inf bounds make it a no-op.
                        total = np.int64(0)
                        exact_mass = -charge_mass_offset
                        rdbe = 1.0
                        for j in range(num_elements):
                            val = c[j] + min_values[j]
                            val_f = np.float64(val)
                            total += val
                            exact_mass += val_f * real_masses[j]
                            rdbe += val_f * rdbe_coeffs[j]

                        if total > 0 and original_min_mass <= exact_mass <= original_max_mass:
                            mass_valid_count += 1
                            # Optional RDBE/octet filter at the leaf
                            store = True
                            if do_rdbe_filter:
                                if rdbe < rdbe_min or rdbe > rdbe_max:
                                    store = False
                                elif check_octet:
                                    # Use integer check instead of float
                                    # remainder to avoid expensive Python
                                    # _rem call emitted by Numba
                                    doubled_int = np.int64(2.0 * rdbe)
                                    is_half_int = (doubled_int & 1) == 1
                                    if charge_parity_even and is_half_int:
                                        store = False
                                    elif not charge_parity_even and not is_half_int:
                                        store = False

                            if store:
                                for j in range(num_elements):
                                    result[count, j] = np.int32(c[j] + min_values[j])
                                count += 1

                    i += 1

                # Check bounds
                while i <= k and c[i] >= bounds[i]:
                    m += c[i] * integer_masses[i]
                    c[i] = 0
                    i += 1
                if i <= k:
                    m -= integer_masses[i]
                    c[i] += 1

    return result[:count]
