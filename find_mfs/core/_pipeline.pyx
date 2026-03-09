# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Compiled query pipeline for FormulaFinder.

This module keeps public APIs unchanged while moving post-decomposition
filtering/orchestration work into Cython.
"""
from __future__ import annotations

import numpy as np
cimport numpy as np


def _apply_mask_to_raw(
    dict raw,
    np.ndarray mask,
):
    """Apply a boolean mask to all aligned arrays inside raw result dict."""
    if mask.size == 0:
        return raw

    if np.all(mask):
        return raw

    raw['counts'] = raw['counts'][mask]
    raw['exact_masses'] = raw['exact_masses'][mask]
    raw['error_ppm'] = raw['error_ppm'][mask]
    raw['error_da'] = raw['error_da'][mask]

    if raw.get('rdbe') is not None:
        raw['rdbe'] = raw['rdbe'][mask]

    if raw.get('iso_rmse') is not None:
        raw['iso_rmse'] = raw['iso_rmse'][mask]
        raw['iso_match_frac'] = raw['iso_match_frac'][mask]
        raw['iso_n_matched'] = raw['iso_n_matched'][mask]

    if raw.get('iso_peak_matches') is not None:
        raw['iso_peak_matches'] = raw['iso_peak_matches'][mask]

    return raw


def _build_ion_counts_and_symbols(
    list core_symbols,
    np.ndarray counts,
    object adduct_elements,
):
    """Build ion-space symbols/counts (core + signed adduct offsets)."""
    if not adduct_elements:
        return list(core_symbols), counts

    adduct_only_symbols = [s for s in adduct_elements if s not in core_symbols]
    ion_symbols = list(core_symbols) + adduct_only_symbols

    cdef int n_results = counts.shape[0]
    cdef int n_core = counts.shape[1]
    cdef int n_ion = len(ion_symbols)

    ion_counts = np.zeros((n_results, n_ion), dtype=np.int32)
    ion_counts[:, :n_core] = counts

    offsets = np.array([adduct_elements.get(s, 0) for s in ion_symbols], dtype=np.int32)
    ion_counts += offsets[np.newaxis, :]

    return ion_symbols, ion_counts


def run_query_pipeline(
    dict raw,
    list core_symbols,
    int charge,
    double query_mass,
    object remaining_filter_rdbe = None,
    bint remaining_check_octet = False,
    object isotope_match = None,
    object adduct_elements = None,
    bint adduct_present = False,
    object unknown_symbol_indices = None,
):
    """
    Apply remaining validation and isotope filtering in compiled code.

    Args:
        raw: Dict from decompose_and_score.
        core_symbols: Element symbols for core formula space.
        charge: Ion charge.
        query_mass: Queried ion m/z used for ppm-to-Da tolerance conversion.
        remaining_filter_rdbe: Optional RDBE range filter to apply.
        remaining_check_octet: Whether octet parity check remains.
        isotope_match: Optional SingleEnvelopeMatch config.
        adduct_elements: Optional signed adduct element offsets.
        adduct_present: Whether an adduct was specified in the query.
        unknown_symbol_indices: Optional element column indices without known
            bond-electron definitions. Candidates with non-zero counts in these
            columns fail residual RDBE/octet validation.
    """
    counts = raw['counts']
    cdef int n_rows = counts.shape[0]

    # Residual RDBE/octet filtering (for cases not pre-filtered in decomposition).
    if n_rows > 0 and (remaining_filter_rdbe is not None or remaining_check_octet):
        rdbe_arr = raw.get('rdbe')
        if rdbe_arr is None:
            mask = np.zeros(n_rows, dtype=bool)
            raw = _apply_mask_to_raw(raw, mask)
            n_rows = 0
        else:
            mask = np.ones(n_rows, dtype=bool)

            if unknown_symbol_indices is not None and len(unknown_symbol_indices) > 0:
                unknown_counts = counts[:, unknown_symbol_indices]
                if unknown_counts.ndim == 1:
                    mask &= (unknown_counts == 0)
                else:
                    mask &= np.all(unknown_counts == 0, axis=1)

            if remaining_filter_rdbe is not None:
                rdbe_min = remaining_filter_rdbe[0]
                rdbe_max = remaining_filter_rdbe[1]
                mask &= (rdbe_arr >= rdbe_min) & (rdbe_arr <= rdbe_max)

            if remaining_check_octet:
                # Octet applies to core formula charge parity:
                # with adduct -> core is neutral; otherwise core carries ion charge.
                core_charge = 0 if adduct_present else charge
                parity_even = (abs(core_charge) % 2) == 0
                doubled = np.rint(2.0 * rdbe_arr).astype(np.int64)
                is_half_integer = (doubled & 1) == 1
                if parity_even:
                    mask &= ~is_half_integer
                else:
                    mask &= is_half_integer

            raw = _apply_mask_to_raw(raw, mask)
            counts = raw['counts']
            n_rows = counts.shape[0]

    # Isotope matching/filtering in one batch call (covers old fused+eager paths).
    if isotope_match is not None:
        n_obs = isotope_match.envelope.shape[0]
        if n_rows == 0:
            raw['iso_rmse'] = np.empty(0, dtype=np.float64)
            raw['iso_match_frac'] = np.empty(0, dtype=np.float64)
            raw['iso_n_matched'] = np.empty(0, dtype=np.int32)
            raw['iso_peak_matches'] = np.empty((0, n_obs), dtype=np.int8)
            return raw

        from ..isotopes._isospec import score_isotope_batch

        ion_symbols, ion_counts = _build_ion_counts_and_symbols(
            core_symbols=core_symbols,
            counts=counts,
            adduct_elements=adduct_elements,
        )

        # Guard against chemically invalid ion compositions (negative counts)
        # when signed adduct offsets remove atoms (e.g. adduct='-H').
        nonneg_mask = np.all(ion_counts >= 0, axis=1)
        if not np.all(nonneg_mask):
            raw = _apply_mask_to_raw(raw, nonneg_mask)
            ion_counts = ion_counts[nonneg_mask]
            counts = raw['counts']
            n_rows = counts.shape[0]
            if n_rows == 0:
                raw['iso_rmse'] = np.empty(0, dtype=np.float64)
                raw['iso_match_frac'] = np.empty(0, dtype=np.float64)
                raw['iso_n_matched'] = np.empty(0, dtype=np.int32)
                raw['iso_peak_matches'] = np.empty((0, n_obs), dtype=np.int8)
                return raw

        ppm_to_da = 1e-6 * (isotope_match.mz_tolerance_ppm or 0.0) * query_mass
        mz_tol = max(isotope_match.mz_tolerance_da or 0.0, ppm_to_da)

        iso_rmse, iso_mf, iso_nm, iso_pm = score_isotope_batch(
            ion_symbols,
            ion_counts,
            charge,
            isotope_match.envelope,
            mz_tol,
            isotope_match.simulated_mz_tolerance,
            isotope_match.simulated_intensity_threshold,
        )

        mask = iso_rmse <= isotope_match.minimum_rmse
        if not np.all(mask):
            raw = _apply_mask_to_raw(raw, mask)
            iso_rmse = iso_rmse[mask]
            iso_mf = iso_mf[mask]
            iso_nm = iso_nm[mask]
            iso_pm = iso_pm[mask]

        raw['iso_rmse'] = iso_rmse
        raw['iso_match_frac'] = iso_mf
        raw['iso_n_matched'] = iso_nm
        raw['iso_peak_matches'] = iso_pm

    return raw
