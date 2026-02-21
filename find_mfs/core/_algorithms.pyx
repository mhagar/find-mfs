# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Cython production decomposition kernel.

Adapted from benchmarks/cython_decompose.pyx (proven correct and 1.3-1.9x
faster than Numba in steady state with 0.2ms vs 793ms cold start).
"""
import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt
from libc.stdlib cimport malloc, free


cdef inline bint _is_decomposable(
    float64_t[:, ::1] ERT,
    int i,
    int64_t m,
    int64_t a1,
) noexcept nogil:
    if m < 0:
        return False
    if ERT[m % a1, i] <= <float64_t>m:
        return True
    return False


cdef int _decompose_core(
    float64_t[:, ::1] ERT,
    int64_t[::1] integer_masses,
    float64_t[::1] real_masses,
    float64_t[::1] bounds,
    int64_t[::1] min_values,
    int64_t min_int,
    int64_t max_int,
    double original_min_mass,
    double original_max_mass,
    double charge_mass_offset,
    int max_results,
    float64_t[::1] rdbe_coeffs,
    double rdbe_min,
    double rdbe_max,
    bint check_octet,
    bint charge_parity_even,
    bint do_rdbe_filter,
    bint do_iso_filter,
    float64_t[::1] iso_m1_coeffs,
    float64_t[::1] iso_m2_direct,
    double obs_m1_ratio,
    double obs_m2_ratio,
    double iso_tol_rel,
    double iso_tol_abs,
    int num_elements,
    int32_t[:, ::1] out_counts,
) noexcept nogil:
    """
    Core decomposition loop. Returns the number of valid decompositions
    written to out_counts.

    This is a cdef function so it can be called from other Cython modules
    (via cimport from _algorithms.pxd) without Python overhead.
    """
    cdef int64_t a1 = integer_masses[0]
    cdef int k = num_elements - 1

    # Working array (stack-allocated would be ideal but num_elements is dynamic)
    cdef int64_t* c = <int64_t*>malloc(num_elements * sizeof(int64_t))
    if c == NULL:
        return 0

    cdef int count = 0
    cdef int mass_valid_count = 0
    cdef int64_t m_target, m
    cdef int i, j
    cdef int64_t total, val
    cdef double val_f, exact_mass, rdbe, approx_m1, approx_m2, tol, tol2
    cdef int64_t doubled_int
    cdef bint is_half_int, store

    for m_target in range(min_int, max_int + 1):
        # Reset state
        for j in range(num_elements):
            c[j] = 0
        i = k
        m = m_target

        while i <= k and mass_valid_count < max_results:
            if not _is_decomposable(ERT, i, m, a1):
                # Backtrack until decomposable
                while i <= k and not _is_decomposable(ERT, i, m, a1):
                    m = m + c[i] * integer_masses[i]
                    c[i] = 0
                    i = i + 1
                # Check bounds
                while i <= k and c[i] >= <int64_t>bounds[i]:
                    m = m + c[i] * integer_masses[i]
                    c[i] = 0
                    i = i + 1
                if i <= k:
                    m = m - integer_masses[i]
                    c[i] = c[i] + 1
            else:
                # Descend as deep as possible
                while i > 0 and _is_decomposable(ERT, i - 1, m, a1):
                    i = i - 1

                if i == 0:
                    c[0] = m // a1

                    # Check bounds for element 0
                    if c[0] <= <int64_t>bounds[0]:
                        total = 0
                        exact_mass = -charge_mass_offset
                        rdbe = 1.0
                        approx_m1 = 0.0
                        for j in range(num_elements):
                            val = c[j] + min_values[j]
                            val_f = <double>val
                            total = total + val
                            exact_mass = exact_mass + val_f * real_masses[j]
                            rdbe = rdbe + val_f * rdbe_coeffs[j]
                            approx_m1 = approx_m1 + val_f * iso_m1_coeffs[j]

                        if total > 0 and original_min_mass <= exact_mass <= original_max_mass:
                            mass_valid_count = mass_valid_count + 1
                            store = True

                            if do_rdbe_filter:
                                if rdbe < rdbe_min or rdbe > rdbe_max:
                                    store = False
                                elif check_octet:
                                    doubled_int = <int64_t>(2.0 * rdbe)
                                    is_half_int = (doubled_int & 1) == 1
                                    if charge_parity_even and is_half_int:
                                        store = False
                                    elif not charge_parity_even and not is_half_int:
                                        store = False

                            # Optional isotope pre-filter
                            if store and do_iso_filter:
                                tol = iso_tol_rel * obs_m1_ratio
                                if tol < iso_tol_abs:
                                    tol = iso_tol_abs
                                if fabs(approx_m1 - obs_m1_ratio) > tol:
                                    store = False
                                elif obs_m2_ratio > 0.0:
                                    approx_m2 = approx_m1 * approx_m1 * 0.5
                                    for j in range(num_elements):
                                        approx_m2 = approx_m2 + <double>(c[j] + min_values[j]) * iso_m2_direct[j]
                                    tol2 = iso_tol_rel * obs_m2_ratio
                                    if tol2 < iso_tol_abs:
                                        tol2 = iso_tol_abs
                                    if fabs(approx_m2 - obs_m2_ratio) > tol2:
                                        store = False

                            if store:
                                for j in range(num_elements):
                                    out_counts[count, j] = <int32_t>(c[j] + min_values[j])
                                count = count + 1

                    i = i + 1

                # Check bounds
                while i <= k and c[i] >= <int64_t>bounds[i]:
                    m = m + c[i] * integer_masses[i]
                    c[i] = 0
                    i = i + 1
                if i <= k:
                    m = m - integer_masses[i]
                    c[i] = c[i] + 1

    free(c)
    return count


def _decompose_mass_range(
    np.ndarray[float64_t, ndim=2] ERT_np,
    np.ndarray[int64_t, ndim=1] integer_masses_np,
    np.ndarray[float64_t, ndim=1] real_masses_np,
    np.ndarray[float64_t, ndim=1] bounds_np,
    np.ndarray[int64_t, ndim=1] min_values_np,
    int64_t min_int,
    int64_t max_int,
    double original_min_mass,
    double original_max_mass,
    double charge_mass_offset,
    int max_results,
    np.ndarray[float64_t, ndim=1] rdbe_coeffs_np,
    double rdbe_min,
    double rdbe_max,
    bint check_octet,
    bint charge_parity_even,
    bint do_rdbe_filter,
    bint do_iso_filter = False,
    np.ndarray[float64_t, ndim=1] iso_m1_coeffs_np = None,
    np.ndarray[float64_t, ndim=1] iso_m2_direct_np = None,
    double obs_m1_ratio = 0.0,
    double obs_m2_ratio = 0.0,
    double iso_tol_rel = 0.3,
    double iso_tol_abs = 0.02,
):
    """
    Production Cython implementation of mass decomposition.

    Drop-in replacement for the Numba @njit _decompose_mass_range.
    Returns 2D int32 array of shape (N, num_elements) with valid counts.
    """
    cdef int num_elements = integer_masses_np.shape[0]

    # Handle optional arrays
    if iso_m1_coeffs_np is None:
        iso_m1_coeffs_np = np.zeros(num_elements, dtype=np.float64)
    if iso_m2_direct_np is None:
        iso_m2_direct_np = np.zeros(num_elements, dtype=np.float64)

    # Pre-allocate output buffer
    cdef np.ndarray[int32_t, ndim=2] result = np.empty(
        (max_results, num_elements), dtype=np.int32,
    )

    # Typed memoryviews
    cdef float64_t[:, ::1] ERT = ERT_np
    cdef int64_t[::1] integer_masses = integer_masses_np
    cdef float64_t[::1] real_masses = real_masses_np
    cdef float64_t[::1] bounds = bounds_np
    cdef int64_t[::1] min_values = min_values_np
    cdef float64_t[::1] rdbe_coeffs = rdbe_coeffs_np
    cdef float64_t[::1] iso_m1_coeffs = iso_m1_coeffs_np
    cdef float64_t[::1] iso_m2_direct = iso_m2_direct_np
    cdef int32_t[:, ::1] result_view = result

    cdef int count

    with nogil:
        count = _decompose_core(
            ERT, integer_masses, real_masses, bounds, min_values,
            min_int, max_int, original_min_mass, original_max_mass,
            charge_mass_offset, max_results,
            rdbe_coeffs, rdbe_min, rdbe_max,
            check_octet, charge_parity_even, do_rdbe_filter,
            do_iso_filter, iso_m1_coeffs, iso_m2_direct,
            obs_m1_ratio, obs_m2_ratio, iso_tol_rel, iso_tol_abs,
            num_elements, result_view,
        )

    return result[:count]


def decompose_and_score(
    np.ndarray[float64_t, ndim=2] ERT_np,
    np.ndarray[int64_t, ndim=1] integer_masses_np,
    np.ndarray[float64_t, ndim=1] real_masses_np,
    np.ndarray[float64_t, ndim=1] bounds_np,
    np.ndarray[int64_t, ndim=1] min_values_np,
    int64_t min_int,
    int64_t max_int,
    double original_min_mass,
    double original_max_mass,
    double charge_mass_offset,
    int max_results,
    np.ndarray[float64_t, ndim=1] rdbe_coeffs_np,
    double rdbe_min,
    double rdbe_max,
    bint check_octet,
    bint charge_parity_even,
    bint do_rdbe_filter,
    bint do_iso_filter,
    np.ndarray[float64_t, ndim=1] iso_m1_coeffs_np,
    np.ndarray[float64_t, ndim=1] iso_m2_direct_np,
    double obs_m1_ratio,
    double obs_m2_ratio,
    double iso_tol_rel,
    double iso_tol_abs,
    double query_mass,
    double adduct_mass,
    bint compute_rdbe,
):
    """
    Fused decomposition + scoring: decompose, compute exact masses, errors,
    RDBE, and sort by |error_ppm| — all in one call.

    Returns dict of numpy arrays:
        counts: int32[N, n_elem]
        exact_masses: float64[N]
        error_ppm: float64[N]
        error_da: float64[N]
        rdbe: float64[N] or None
    All pre-sorted by |error_ppm|.
    """
    cdef int num_elements = integer_masses_np.shape[0]

    # Pre-allocate output buffer for decomposition
    cdef np.ndarray[int32_t, ndim=2] raw_counts = np.empty(
        (max_results, num_elements), dtype=np.int32,
    )

    # Typed memoryviews for input
    cdef float64_t[:, ::1] ERT = ERT_np
    cdef int64_t[::1] integer_masses = integer_masses_np
    cdef float64_t[::1] real_masses = real_masses_np
    cdef float64_t[::1] bounds = bounds_np
    cdef int64_t[::1] min_values = min_values_np
    cdef float64_t[::1] rdbe_coeffs = rdbe_coeffs_np
    cdef float64_t[::1] iso_m1_coeffs = iso_m1_coeffs_np
    cdef float64_t[::1] iso_m2_direct = iso_m2_direct_np
    cdef int32_t[:, ::1] raw_view = raw_counts

    cdef int n_results
    cdef int i, j

    # Step 1: Decompose
    with nogil:
        n_results = _decompose_core(
            ERT, integer_masses, real_masses, bounds, min_values,
            min_int, max_int, original_min_mass, original_max_mass,
            charge_mass_offset, max_results,
            rdbe_coeffs, rdbe_min, rdbe_max,
            check_octet, charge_parity_even, do_rdbe_filter,
            do_iso_filter, iso_m1_coeffs, iso_m2_direct,
            obs_m1_ratio, obs_m2_ratio, iso_tol_rel, iso_tol_abs,
            num_elements, raw_view,
        )

    if n_results == 0:
        return {
            'counts': np.empty((0, num_elements), dtype=np.int32),
            'exact_masses': np.empty(0, dtype=np.float64),
            'error_ppm': np.empty(0, dtype=np.float64),
            'error_da': np.empty(0, dtype=np.float64),
            'rdbe': None if not compute_rdbe else np.empty(0, dtype=np.float64),
        }

    # Step 2: Compute exact masses, errors, RDBE
    cdef np.ndarray[float64_t, ndim=1] exact_masses = np.empty(n_results, dtype=np.float64)
    cdef np.ndarray[float64_t, ndim=1] error_ppm = np.empty(n_results, dtype=np.float64)
    cdef np.ndarray[float64_t, ndim=1] error_da = np.empty(n_results, dtype=np.float64)
    cdef np.ndarray[float64_t, ndim=1] rdbe_arr
    cdef np.ndarray[float64_t, ndim=1] abs_err = np.empty(n_results, dtype=np.float64)

    if compute_rdbe:
        rdbe_arr = np.empty(n_results, dtype=np.float64)
    else:
        rdbe_arr = np.empty(1, dtype=np.float64)  # dummy for unconditional memoryview bind

    cdef float64_t[::1] exact_masses_v = exact_masses
    cdef float64_t[::1] error_ppm_v = error_ppm
    cdef float64_t[::1] error_da_v = error_da
    cdef float64_t[::1] rdbe_v = rdbe_arr  # always bound to eliminate nogil guard
    cdef float64_t[::1] abs_err_v = abs_err

    cdef double em, ep, ed, r_val, val_f
    cdef int32_t cnt

    with nogil:
        for i in range(n_results):
            em = -charge_mass_offset
            r_val = 1.0
            for j in range(num_elements):
                cnt = raw_view[i, j]
                val_f = <double>cnt
                em = em + val_f * real_masses[j]
            em = em + adduct_mass
            exact_masses_v[i] = em
            ed = em - query_mass
            error_da_v[i] = ed
            ep = ed / query_mass * 1e6
            error_ppm_v[i] = ep
            abs_err_v[i] = fabs(ep)

            if compute_rdbe:
                r_val = 1.0
                for j in range(num_elements):
                    r_val = r_val + <double>raw_view[i, j] * rdbe_coeffs[j]
                rdbe_v[i] = r_val

    # Step 3: Sort by |error_ppm| using numpy argsort
    cdef np.ndarray sort_order = np.argsort(abs_err)

    # Apply sort order
    cdef np.ndarray[int32_t, ndim=2] sorted_counts = raw_counts[:n_results][sort_order]
    cdef np.ndarray[float64_t, ndim=1] sorted_masses = exact_masses[sort_order]
    cdef np.ndarray[float64_t, ndim=1] sorted_ppm = error_ppm[sort_order]
    cdef np.ndarray[float64_t, ndim=1] sorted_da = error_da[sort_order]

    result = {
        'counts': sorted_counts,
        'exact_masses': sorted_masses,
        'error_ppm': sorted_ppm,
        'error_da': sorted_da,
    }

    if compute_rdbe:
        result['rdbe'] = rdbe_arr[sort_order]
    else:
        result['rdbe'] = None

    return result
