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

    return raw


def run_query_pipeline(
    dict raw,
    list core_symbols,
    int charge,
    double query_mass,
    object remaining_filter_rdbe = None,
    bint remaining_check_octet = False,
    object adduct_elements = None,
    bint adduct_present = False,
    object unknown_symbol_indices = None,
):
    """
    Apply remaining RDBE/octet validation in compiled code.

    Isotope scoring is no longer part of this pipeline: candidates are never
    omitted for isotope reasons here. Isotope/mass likelihoods are
    computed post-hoc by `FormulaScorer.score()`.

    Args:
        raw: Dict from decompose_and_score.
        core_symbols: Element symbols for core formula space.
        charge: Ion charge.
        query_mass: Queried ion m/z (retained for signature stability).
        remaining_filter_rdbe: Optional RDBE range filter to apply.
        remaining_check_octet: Whether octet parity check remains.
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

    return raw
