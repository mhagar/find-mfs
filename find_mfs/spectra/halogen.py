"""Halogen (Cl/Br) detection from isotope envelope intensity patterns."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .envelopes import SpectrumArray, ISOTOPE_SPACING


def detect_halogen_envelopes(
    spec_arr: SpectrumArray,
    envelope_labels: NDArray[np.intp],
    spacing: float = ISOTOPE_SPACING,
    tol: float = 0.01,
    m2_m0_threshold: float = 0.32,
) -> NDArray[np.bool_]:
    """
    Detect Cl/Br zig-zag pattern in isotope envelopes.

    Halogens with significant M+2 isotopes (Cl-37, Br-81) produce a
    characteristic pattern where M2 intensity is elevated relative to M1.
    For a single Cl atom, M2/M0 ~ 0.33; for Br, M2/M0 ~ 0.97.

    Detection criteria per envelope:
    - At least 3 peaks (need M0, M1, M2)
    - M2/M0 > m2_m0_threshold
    - M2 > M1 (the zig-zag)

    Returns bool array of shape (n_envelopes,). True = halogenated.
    """
    n_envelopes = int(envelope_labels.max()) + 1 if len(envelope_labels) > 0 else 0
    flags = np.zeros(n_envelopes, dtype=bool)

    for eid in range(n_envelopes):
        peaks = spec_arr[envelope_labels == eid]
        if len(peaks) < 3:
            continue

        # Sort by m/z, M0 is the monoisotopic (lowest)
        order = np.argsort(peaks['mz'])
        mzs = peaks['mz'][order]
        intsys = peaks['intsy'][order]

        m0_mz = mzs[0]
        m0_intsy = intsys[0]

        # Find M1 and M2 by spacing from M0
        m1_mask = np.abs(mzs - (m0_mz + spacing)) <= tol
        m2_mask = np.abs(mzs - (m0_mz + 2 * spacing)) <= tol

        if not np.any(m1_mask) or not np.any(m2_mask):
            continue

        m1_intsy = intsys[m1_mask][0]
        m2_intsy = intsys[m2_mask][0]

        if m0_intsy <= 0:
            continue

        m2_mz = mzs[m2_mask][0]
        # Mass defect trick TODO its hacky
        if abs(m0_mz - m2_mz) > 2.0:
            continue

        if (m2_intsy / m0_intsy) > m2_m0_threshold and m2_intsy > m1_intsy:
            flags[eid] = True

    return flags
