"""
Isotope data bridge for IsoSpecPy.

Provides:
- Per-element isotope data arrays from IsoSpecPy.PeriodicTbl
- M+1/M+2 approximation coefficients for pre-filtering
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Isotope data arrays from IsoSpecPy.PeriodicTbl
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# M+1 / M+2 approximation coefficients for pre-filter
# ---------------------------------------------------------------------------

# M+1 ratio: probability of a single atom contributing a +1 neutron isotope
# relative to the monoisotopic peak. These are P(M+1)/P(M) per atom.
M1_RATIOS: dict[str, float] = {
    'C': 0.010816, 'H': 0.000115, 'N': 0.003654,
    'O': 0.000381, 'P': 0.0, 'S': 0.007895,
    'F': 0.0, 'Cl': 0.003200, 'Br': 0.009700, 'I': 0.0,
    'Si': 0.050800, 'Se': 0.0,
}

# M+2 "direct" ratio: contribution of a single atom to M+2 that is NOT
# accounted for by (M+1)^2/2. This captures elements with significant
# M+2 isotopes (like Cl-37, S-34, etc.).
M2_DIRECT: dict[str, float] = {
    'C': 0.0, 'H': 0.0, 'N': 0.0,
    'O': 0.002055, 'P': 0.0, 'S': 0.044742,
    'F': 0.0, 'Cl': 0.320300, 'Br': 0.970300, 'I': 0.0,
    'Si': 0.033600, 'Se': 0.0,
}
