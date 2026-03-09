# cython: language_level=3
"""
Declaration file for _algorithms.pyx — exposes cdef functions
for cross-module cimport (e.g. from _pipeline.pyx).
"""
from libc.stdint cimport int64_t, int32_t

ctypedef double float64_t

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
) noexcept nogil
