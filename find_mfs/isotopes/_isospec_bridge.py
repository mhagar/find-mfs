"""
Isotope data bridge for IsoSpecPy.

Provides per-element isotope data arrays from IsoSpecPy.PeriodicTbl
for use by the Cython _isospec scoring module.
"""
from __future__ import annotations

import numpy as np


_isotope_cache: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}


def _build_isotope_cache():
    """Build per-element isotope data from IsoSpecPy's PeriodicTbl."""
    if _isotope_cache:
        return
    try:
        from IsoSpecPy import PeriodicTbl
    except ImportError as e:
        raise ImportError(
            "IsoSpecPy is required for isotope data. "
            "Install with: pip install IsoSpecPy"
        ) from e

    for symbol in PeriodicTbl.symbol_to_masses:
        masses = np.array(PeriodicTbl.symbol_to_masses[symbol], dtype=np.float64)
        probs = np.array(PeriodicTbl.symbol_to_probs[symbol], dtype=np.float64)
        _isotope_cache[symbol] = (len(masses), masses, probs)


def get_isotope_arrays(
    symbols: list[str] | tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build flat isotope data arrays for setupIso from element symbols.

    Args:
        symbols: Element symbols (e.g., ['C', 'H', 'N', 'O', 'P', 'S'])

    Returns:
        Tuple of:
        - iso_numbers: int32 array of isotope counts per element
        - flat_masses: float64 array of all isotope masses (concatenated)
        - flat_probs: float64 array of all isotope probabilities (concatenated)
    """
    _build_isotope_cache()

    iso_numbers = np.empty(len(symbols), dtype=np.int32)
    all_masses = []
    all_probs = []

    for i, sym in enumerate(symbols):
        n, masses, probs = _isotope_cache[sym]
        iso_numbers[i] = n
        all_masses.append(masses)
        all_probs.append(probs)

    flat_masses = np.concatenate(all_masses)
    flat_probs = np.concatenate(all_probs)
    return iso_numbers, flat_masses, flat_probs
