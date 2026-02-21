# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Cython IsoSpec bridge — replaces the Numba ctypes path with C-level
function pointer calls via dlopen/dlsym at module init.
"""
import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt
from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t
from libc.string cimport memcpy
from cython.parallel cimport prange
from posix.dlfcn cimport dlopen, dlsym, dlclose, dlerror, RTLD_LAZY

ctypedef double float64_t

# C function pointer types matching IsoSpec's C API
ctypedef void* (*setupIso_t)(int, const int32_t*, const int32_t*, const double*, const double*) noexcept nogil
ctypedef void* (*setupThresholdFixedEnvelope_t)(void*, double, bint, bint) noexcept nogil
ctypedef size_t (*confs_noFixedEnvelope_t)(void*) noexcept nogil
ctypedef const double* (*massesFixedEnvelope_t)(void*) noexcept nogil
ctypedef const double* (*probsFixedEnvelope_t)(void*) noexcept nogil
ctypedef void (*deleteFixedEnvelope_t)(void*, bint) noexcept nogil
ctypedef void (*deleteIso_t)(void*) noexcept nogil
ctypedef void (*freeReleasedArray_t)(void*) noexcept nogil

# Module-level function pointers
cdef setupIso_t _setupIso = NULL
cdef setupThresholdFixedEnvelope_t _setupThreshold = NULL
cdef confs_noFixedEnvelope_t _confs_no = NULL
cdef massesFixedEnvelope_t _getMasses = NULL
cdef probsFixedEnvelope_t _getProbs = NULL
cdef deleteFixedEnvelope_t _deleteFE = NULL
cdef deleteIso_t _deleteIso = NULL
cdef freeReleasedArray_t _freeArray = NULL
cdef void* _lib_handle = NULL
cdef bint _loaded = False

# Module-level cache for isotope arrays keyed by symbol tuple
_iso_array_cache = {}


def _load_isospec_lib():
    """Load IsoSpec shared library and resolve function pointers."""
    global _setupIso, _setupThreshold, _confs_no
    global _getMasses, _getProbs, _deleteFE, _deleteIso, _freeArray
    global _lib_handle, _loaded

    if _loaded:
        return

    # Get the path to IsoSpec's shared library
    try:
        from IsoSpecPy.isoFFI import isoFFI
        lib_path = str(isoFFI.libpath)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Cannot find IsoSpecPy C++ library: {e}. "
            "Install IsoSpecPy with: pip install IsoSpecPy"
        ) from e

    # dlopen the library
    cdef bytes path_bytes = lib_path.encode('utf-8')
    _lib_handle = dlopen(path_bytes, RTLD_LAZY)
    if _lib_handle == NULL:
        err = dlerror()
        raise ImportError(
            f"Cannot load IsoSpecPy C++ library at {lib_path}: "
            f"{err.decode('utf-8') if err else 'unknown error'}"
        )

    # Resolve function pointers
    _setupIso = <setupIso_t>dlsym(_lib_handle, "setupIso")
    _setupThreshold = <setupThresholdFixedEnvelope_t>dlsym(_lib_handle, "setupThresholdFixedEnvelope")
    _confs_no = <confs_noFixedEnvelope_t>dlsym(_lib_handle, "confs_noFixedEnvelope")
    _getMasses = <massesFixedEnvelope_t>dlsym(_lib_handle, "massesFixedEnvelope")
    _getProbs = <probsFixedEnvelope_t>dlsym(_lib_handle, "probsFixedEnvelope")
    _deleteFE = <deleteFixedEnvelope_t>dlsym(_lib_handle, "deleteFixedEnvelope")
    _deleteIso = <deleteIso_t>dlsym(_lib_handle, "deleteIso")
    _freeArray = <freeReleasedArray_t>dlsym(_lib_handle, "freeReleasedArray")

    # Verify all resolved
    if (_setupIso == NULL or _setupThreshold == NULL or _confs_no == NULL or
        _getMasses == NULL or _getProbs == NULL or _deleteFE == NULL or
        _deleteIso == NULL or _freeArray == NULL):
        dlclose(_lib_handle)
        _lib_handle = NULL
        raise ImportError("Failed to resolve one or more IsoSpec C functions")

    _loaded = True


ctypedef signed char int8_t


cdef (double, double, int) _score_single_envelope(
    int32_t* iso_numbers, int32_t* atom_counts,
    double* flat_masses, double* flat_probs,
    int n_elements,
    double* obs_mz, double* obs_int, int n_obs,
    double combine_tol, double match_tol, double threshold,
    int charge, double electron_mass,
    int8_t* peak_matches_out,
) noexcept nogil:
    """
    Score a single candidate against the observed envelope.
    All C-level, no GIL, no Python objects.

    If peak_matches_out is not NULL, writes per-observed-peak match
    booleans (1=matched, 0=unmatched) into it.

    Returns (rmse, match_fraction, n_matched).
    """
    cdef void* iso_ptr
    cdef void* env_ptr
    cdef size_t n_peaks
    cdef const double* masses_raw
    cdef const double* probs_raw

    # Zero out peak_matches if provided
    cdef int ii
    if peak_matches_out != NULL:
        for ii in range(n_obs):
            peak_matches_out[ii] = 0

    # 1. Call C++ setupIso
    iso_ptr = _setupIso(
        n_elements, iso_numbers, atom_counts,
        flat_masses, flat_probs,
    )

    # 2. Get threshold fixed envelope
    env_ptr = _setupThreshold(iso_ptr, threshold, False, False)
    n_peaks = _confs_no(env_ptr)

    if n_peaks == 0:
        _deleteFE(env_ptr, False)
        _deleteIso(iso_ptr)
        return (1.0, 0.0, 0)

    # 3. Read mass/prob arrays
    masses_raw = _getMasses(env_ptr)
    probs_raw = _getProbs(env_ptr)

    # Copy to local C arrays (C++ memory will be freed)
    cdef double* pred_mz = <double*>malloc(n_peaks * sizeof(double))
    cdef double* pred_prob = <double*>malloc(n_peaks * sizeof(double))
    if pred_mz == NULL or pred_prob == NULL:
        if pred_mz != NULL: free(pred_mz)
        if pred_prob != NULL: free(pred_prob)
        _freeArray(<void*>masses_raw)
        _freeArray(<void*>probs_raw)
        _deleteFE(env_ptr, False)
        _deleteIso(iso_ptr)
        return (1.0, 0.0, 0)

    cdef size_t i, j
    for i in range(n_peaks):
        pred_mz[i] = masses_raw[i]
        pred_prob[i] = probs_raw[i]

    # 4. Free C++ memory
    _freeArray(<void*>masses_raw)
    _freeArray(<void*>probs_raw)
    _deleteFE(env_ptr, False)
    _deleteIso(iso_ptr)

    # 5. Adjust for charge (convert neutral mass to m/z)
    cdef int abs_charge
    cdef double charge_offset
    if charge != 0:
        abs_charge = abs(charge)
        charge_offset = charge * electron_mass
        for i in range(n_peaks):
            pred_mz[i] = (pred_mz[i] - charge_offset) / abs_charge

    # 6. Insertion sort by mass
    cdef double key_mz, key_prob
    cdef int ki
    for ki in range(1, <int>n_peaks):
        key_mz = pred_mz[ki]
        key_prob = pred_prob[ki]
        j = ki - 1
        while <int>j >= 0 and pred_mz[j] > key_mz:
            pred_mz[j + 1] = pred_mz[j]
            pred_prob[j + 1] = pred_prob[j]
            j -= 1
        pred_mz[j + 1] = key_mz
        pred_prob[j + 1] = key_prob

    # 7. Combine unresolved isotopologues
    cdef double* combined_mz = <double*>malloc(n_peaks * sizeof(double))
    cdef double* combined_int = <double*>malloc(n_peaks * sizeof(double))
    if combined_mz == NULL or combined_int == NULL:
        free(pred_mz)
        free(pred_prob)
        if combined_mz != NULL: free(combined_mz)
        if combined_int != NULL: free(combined_int)
        return (1.0, 0.0, 0)

    cdef int n_combined = 0
    cdef double grp_mz_sum, grp_int_sum

    i = 0
    while i < n_peaks:
        grp_mz_sum = pred_mz[i] * pred_prob[i]
        grp_int_sum = pred_prob[i]
        j = i + 1
        while j < n_peaks and fabs(pred_mz[j] - pred_mz[i]) <= combine_tol:
            grp_mz_sum += pred_mz[j] * pred_prob[j]
            grp_int_sum += pred_prob[j]
            j += 1
        combined_mz[n_combined] = grp_mz_sum / grp_int_sum
        combined_int[n_combined] = grp_int_sum
        n_combined += 1
        i = j

    free(pred_mz)
    free(pred_prob)

    # 8. Rescale to base peak = 1.0
    cdef double mx = 0.0
    for i in range(<size_t>n_combined):
        if combined_int[i] > mx:
            mx = combined_int[i]
    if mx > 0.0:
        for i in range(<size_t>n_combined):
            combined_int[i] /= mx

    # 9. Match observed peaks to closest predicted
    cdef double best_diff, d_val, pred_val, rmse, match_frac
    cdef int best_j, n_matched = 0
    cdef double sse = 0.0
    cdef int count = 0, base_idx = 0
    cdef double max_obs = obs_int[0]
    cdef bint matched

    # Find base peak index
    for i in range(1, <size_t>n_obs):
        if obs_int[i] > max_obs:
            max_obs = obs_int[i]
            base_idx = <int>i

    # Match and score
    for i in range(<size_t>n_obs):
        best_diff = 1e30
        best_j = -1
        for j in range(<size_t>n_combined):
            d_val = fabs(obs_mz[i] - combined_mz[j])
            if d_val < best_diff:
                best_diff = d_val
                best_j = <int>j

        pred_val = 0.0
        matched = False
        if best_diff <= match_tol:
            pred_val = combined_int[best_j]
            matched = True
            n_matched += 1

        if peak_matches_out != NULL:
            peak_matches_out[i] = 1 if matched else 0

        if <int>i != base_idx:
            d_val = obs_int[i] - pred_val
            sse += d_val * d_val
            count += 1

    free(combined_mz)
    free(combined_int)

    rmse = sqrt(sse / count) if count > 0 else 0.0
    match_frac = <double>n_matched / <double>n_obs if n_obs > 0 else 0.0

    return (rmse, match_frac, n_matched)


cdef (double, double, int) _score_candidate_zeroskip(
    int32_t* iso_numbers_ptr, double* flat_masses_ptr, double* flat_probs_ptr,
    int* iso_offsets, int n_elements,
    int32_t* candidate_counts,
    double* obs_mz_ptr, double* obs_int_ptr, int n_obs,
    double combine_tol, double match_tol, double threshold,
    int charge, double electron_mass,
    int8_t* peak_matches_out,
) noexcept nogil:
    """Score a single candidate, skipping zero-count elements."""
    cdef int j, k, n_active, n_iso, offset
    cdef int32_t* active_iso_numbers
    cdef int32_t* active_counts
    cdef double* active_masses
    cdef double* active_probs
    cdef double r, mf
    cdef int nm

    # Count non-zero elements
    n_active = 0
    for j in range(n_elements):
        if candidate_counts[j] > 0:
            n_active += 1

    if n_active == 0:
        if peak_matches_out != NULL:
            for j in range(n_obs):
                peak_matches_out[j] = 0
        return (1.0, 0.0, 0)

    # Build compressed arrays with only non-zero elements
    active_iso_numbers = <int32_t*>malloc(n_active * sizeof(int32_t))
    active_counts = <int32_t*>malloc(n_active * sizeof(int32_t))

    k = 0
    n_iso = 0
    for j in range(n_elements):
        if candidate_counts[j] > 0:
            active_iso_numbers[k] = iso_numbers_ptr[j]
            active_counts[k] = candidate_counts[j]
            n_iso += iso_numbers_ptr[j]
            k += 1

    active_masses = <double*>malloc(n_iso * sizeof(double))
    active_probs = <double*>malloc(n_iso * sizeof(double))

    # Copy flat mass/prob data for active elements using precomputed offsets
    offset = 0
    for j in range(n_elements):
        if candidate_counts[j] > 0:
            memcpy(&active_masses[offset], &flat_masses_ptr[iso_offsets[j]],
                   iso_numbers_ptr[j] * sizeof(double))
            memcpy(&active_probs[offset], &flat_probs_ptr[iso_offsets[j]],
                   iso_numbers_ptr[j] * sizeof(double))
            offset += iso_numbers_ptr[j]

    r, mf, nm = _score_single_envelope(
        active_iso_numbers, active_counts,
        active_masses, active_probs, n_active,
        obs_mz_ptr, obs_int_ptr, n_obs,
        combine_tol, match_tol, threshold,
        charge, electron_mass,
        peak_matches_out,
    )

    free(active_iso_numbers)
    free(active_counts)
    free(active_masses)
    free(active_probs)

    return (r, mf, nm)


def score_isotope_batch(
    list symbols,
    np.ndarray counts_2d,
    int charge,
    np.ndarray observed_envelope,
    double mz_match_tolerance,
    double simulated_mz_tolerance = 0.05,
    double simulated_intensity_threshold = 0.001,
):
    """
    Batch isotope envelope scoring for multiple candidates.

    Uses OpenMP prange for parallel scoring and skips zero-count elements
    to reduce IsoSpec setup overhead.

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
    _load_isospec_lib()

    from molmass.elements import ELECTRON

    cdef int n_elements = len(symbols)
    counts_2d = np.ascontiguousarray(counts_2d, dtype=np.int32)
    cdef int n_candidates = counts_2d.shape[0]

    # Cache isotope arrays keyed by symbol tuple
    sym_key = tuple(symbols)
    if sym_key in _iso_array_cache:
        iso_numbers_np, flat_masses_np, flat_probs_np = _iso_array_cache[sym_key]
    else:
        from ._isospec_bridge import get_isotope_arrays
        iso_numbers_np, flat_masses_np, flat_probs_np = get_isotope_arrays(symbols)
        _iso_array_cache[sym_key] = (iso_numbers_np, flat_masses_np, flat_probs_np)

    cdef np.ndarray iso_numbers = np.ascontiguousarray(iso_numbers_np, dtype=np.int32)
    cdef np.ndarray flat_masses = np.ascontiguousarray(flat_masses_np, dtype=np.float64)
    cdef np.ndarray flat_probs = np.ascontiguousarray(flat_probs_np, dtype=np.float64)

    cdef np.ndarray obs_mz = np.ascontiguousarray(observed_envelope[:, 0], dtype=np.float64)
    cdef np.ndarray obs_int = np.ascontiguousarray(observed_envelope[:, 1], dtype=np.float64)
    cdef int n_obs = obs_mz.shape[0]

    cdef double electron_mass = ELECTRON.mass

    # Output arrays
    cdef np.ndarray rmse_out = np.empty(n_candidates, dtype=np.float64)
    cdef np.ndarray mf_out = np.empty(n_candidates, dtype=np.float64)
    cdef np.ndarray nm_out = np.empty(n_candidates, dtype=np.int32)
    cdef np.ndarray peak_matches_out = np.empty((n_candidates, n_obs), dtype=np.int8)

    cdef int i, j
    cdef double r, mf
    cdef int nm

    # Get raw pointers for nogil block via memoryviews
    cdef int32_t[::1] iso_numbers_view = iso_numbers
    cdef double[::1] flat_masses_view = flat_masses
    cdef double[::1] flat_probs_view = flat_probs
    cdef double[::1] obs_mz_view = obs_mz
    cdef double[::1] obs_int_view = obs_int
    cdef int32_t[:, ::1] counts_view = counts_2d
    cdef double[::1] rmse_out_view = rmse_out
    cdef double[::1] mf_out_view = mf_out
    cdef int32_t[::1] nm_out_view = nm_out
    cdef int8_t[:, ::1] pm_out_view = peak_matches_out

    # Pre-compute cumulative offsets into flat isotope arrays for zero-skip
    cdef int* iso_offsets = <int*>malloc((n_elements + 1) * sizeof(int))
    if iso_offsets == NULL:
        raise MemoryError("Failed to allocate iso_offsets")
    iso_offsets[0] = 0
    for j in range(n_elements):
        iso_offsets[j + 1] = iso_offsets[j] + iso_numbers_view[j]
    cdef int total_isotopes = iso_offsets[n_elements]

    # Pointers to flat arrays for nogil access
    cdef int32_t* iso_numbers_ptr = &iso_numbers_view[0]
    cdef double* flat_masses_ptr = &flat_masses_view[0]
    cdef double* flat_probs_ptr = &flat_probs_view[0]
    cdef double* obs_mz_ptr = &obs_mz_view[0]
    cdef double* obs_int_ptr = &obs_int_view[0]

    with nogil:
        for i in prange(n_candidates, schedule='dynamic', chunksize=4):
            r, mf, nm = _score_candidate_zeroskip(
                iso_numbers_ptr, flat_masses_ptr, flat_probs_ptr,
                iso_offsets, n_elements,
                &counts_view[i, 0],
                obs_mz_ptr, obs_int_ptr, n_obs,
                simulated_mz_tolerance, mz_match_tolerance,
                simulated_intensity_threshold,
                charge, electron_mass,
                &pm_out_view[i, 0],
            )
            rmse_out_view[i] = r
            mf_out_view[i] = mf
            nm_out_view[i] = nm

    free(iso_offsets)
    return rmse_out, mf_out, nm_out, peak_matches_out

