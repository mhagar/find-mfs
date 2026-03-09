"""
Per-atom M+1 and M+2 isotope abundance ratios, retrieved from molmass

Used by the approximate isotope pre-filter in the decomposition kernel
to quickly reject candidates whose predicted M+1/M+2 envelope ratios
are far from the observed spectrum
"""
from __future__ import annotations

from molmass import ELEMENTS

_m1m2_cache: dict[str, tuple[float, float]] = {}


def _compute_m1m2_ratio(symbol: str) -> tuple[float, float]:
    """
    Derive per-atom M+1 and M+2 ratios from molmass isotope data
    """
    if symbol in _m1m2_cache:
        return _m1m2_cache[symbol]

    isotopes = ELEMENTS[symbol].isotopes
    mono_mn = min(isotopes.keys())
    mono_ab = isotopes[mono_mn].abundance

    m1 = 0.0
    m2 = 0.0
    for mn, iso in isotopes.items():
        delta = mn - mono_mn
        if delta == 1:
            m1 += iso.abundance / mono_ab
        elif delta == 2:
            m2 += iso.abundance / mono_ab

    _m1m2_cache[symbol] = (m1, m2)
    return m1, m2


def get_m1_ratio(symbol: str) -> float:
    """
    M+1/M+0 abundance ratio per atom of the given element
    """
    return _compute_m1m2_ratio(symbol)[0]


def get_m2_direct(symbol: str) -> float:
    """
    M+2/M+0 abundance ratio per atom (direct, not from (M+1)^2/2)
    """
    return _compute_m1m2_ratio(symbol)[1]
