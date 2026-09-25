"""
Halogen (Cl/Br) detection from isotope envelope intensity patterns
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .envelopes import SpectrumArray, ISOTOPE_SPACING

# M+2 mass offsets of the heavy halogen isotopes: 37Cl/35Cl and 81Br/79Br
# The M+2 search is centred between them.
# Note: 2 * ISOTOPE_SPACING (2.0067 Da, the 13C2 position) is bad,
#   sits ~10 mDa away from both.
CL_M2_OFFSET = 1.99705
BR_M2_OFFSET = 1.99795
HALOGEN_M2_OFFSET = (CL_M2_OFFSET + BR_M2_OFFSET) / 2

# One Cl gives M2/M0 ~ 0.32 on top of the non-halogen M+2
# One Br ~ 0.97.
# Sulfur is the main non-halogen source of excess M+2 (34S: ~0.045 per S),
# so 0.2 leaves room for ~S4 before a false positive
DEFAULT_MIN_M2_EXCESS = 0.2


def detect_halogen_envelopes(
    spec_arr: SpectrumArray,
    envelope_labels: NDArray[np.intp],
    charge: int = 1,
    tol: float = 0.02,
    ppm: float = 0.0,
    min_m2_excess: float = DEFAULT_MIN_M2_EXCESS,
) -> NDArray[np.bool_]:
    """
    Flag each labelled isotope envelope as Cl/Br-containing or not.
    See `envelope_is_halogen` for the criteria.

    Returns bool array of shape (n_envelopes,). True = halogenated.
    """
    n_envelopes = int(envelope_labels.max()) + 1 if len(envelope_labels) > 0 else 0
    flags = np.zeros(n_envelopes, dtype=bool)

    for eid in range(n_envelopes):
        peaks = spec_arr[envelope_labels == eid]
        flags[eid] = envelope_is_halogen(
            peaks, charge=charge, tol=tol, ppm=ppm, min_m2_excess=min_m2_excess,
        )
    return flags


def expected_nonhalogen_m2(
        m1_m0: float
) -> float:
    """
    Upper estimate of M2/M0 for a halogen-free ion, given its observed M1/M0.

    Treating M+1 as coming from n independent heavy-isotope sites (mostly 13C),
    the binomial M2/M0 is (M1/M0)^2 * (n-1)/(2n), which is always below
    (M1/M0)^2 / 2.

    Using the bound keeps large carbon-rich ions from looking halogenated.
    18O/34S M+2 is not predicted from M+1; it is what `min_m2_excess` leaves room for.
    """
    return m1_m0 ** 2 / 2


def envelope_is_halogen(
        envelope: SpectrumArray | NDArray,
        charge: int = 1,
        tol: float = 0.02,
        ppm: float = 0.0,
        min_m2_excess: float = DEFAULT_MIN_M2_EXCESS,
) -> bool:
    """
    Given an isotope envelope as an array, returns True if its
     M+2 is too tall to be explained without Cl/Br.

    The lowest-m/z peak is taken as M0, so the envelope must not contain
    anything below the monoisotopic peak.

    Detection criteria:
    - An M+2 peak near the Cl/Br offset (M0 + ~1.9975/z)
    - M2/M0 must exceed the non-halogen M+2 predicted from M1/M0
        (see `expected_nonhalogen_m2`) by more than `min_m2_excess`

    M+1 may legitimately be absent (i.e. per-brominated ions, weak signal)
    so a missing M+1 counts as zero.

    Peaks within the tolerance of M+1 / M+2 are summed,
     so an M+2 resolved into 37Cl and 13C2 fine structure is handled.

    Args:
        envelope: Peaks of one isotope envelope.
        charge: Charge state; isotope offsets are divided by it.
        tol: Minimum m/z tolerance (Da) for locating M+1 / M+2.
        ppm: Mass accuracy of the data. Widens the tolerance to 3 sigma of the
            error on an M0-M2 difference, when that exceeds `tol`.
        min_m2_excess: How far M2/M0 must exceed the non-halogen prediction.
    """
    envelope = np.asarray(envelope)
    if len(envelope) < 2:
        return False

    mzs = envelope['mz']
    intsys = envelope['intsy']

    # M0 is the monoisotopic (lowest-m/z) peak
    m0 = int(np.argmin(mzs))
    m0_mz = float(mzs[m0])
    m0_intsy = float(intsys[m0])
    if m0_intsy <= 0:
        return False

    z = abs(charge) or 1
    window = max(tol, 3 * np.sqrt(2) * ppm * 1e-6 * m0_mz)

    m1_mask = np.abs(mzs - (m0_mz + ISOTOPE_SPACING / z)) <= window
    m2_mask = np.abs(mzs - (m0_mz + HALOGEN_M2_OFFSET / z)) <= window

    # M+2 is the halogen signature and is required.
    if not np.any(m2_mask):
        return False

    m1_m0 = float(intsys[m1_mask].sum()) / m0_intsy
    m2_m0 = float(intsys[m2_mask].sum()) / m0_intsy

    return m2_m0 - expected_nonhalogen_m2(m1_m0) > min_m2_excess
