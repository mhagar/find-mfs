"""
Tests for halogen detection
"""
import numpy as np
from molmass import Formula
from find_mfs.spectra.halogen import envelope_is_halogen
from find_mfs.isotopes.envelope import get_isotope_envelope
from find_mfs.spectra.envelopes import to_spec_arr, ISOTOPE_SPACING

THIAMPHENICOL = "C12H15Cl2NO5S"

class TestHalogenDetection:
    def test_envelope_is_halogen(self):
        thiamphenicol = Formula(THIAMPHENICOL)
        envelope = get_isotope_envelope(
            thiamphenicol,
            mz_tolerance=0.05,
            threshold=0.001,
        )

        spec_arr = to_spec_arr(
            envelope[:,0],
            envelope[:,1],
        )

        assert envelope_is_halogen(spec_arr), \
            "Thiamphenicol envelope should be identified as chlorinated"

    def test_bromine_detected_without_m1(self):
        """Br has a strong M+2 but a weak M+1 (13C) that can fall below a noise
        floor. Detection must not require M+1 to be present."""
        env = get_isotope_envelope(
            Formula("C10H10Br2"), mz_tolerance=0.02, threshold=0.001
        )
        env = env[np.argsort(env[:, 0])]
        m0 = env[0, 0]
        # Drop the weak M+1 (13C), keep M0 and the strong Br M+2.
        keep = np.abs(env[:, 0] - (m0 + ISOTOPE_SPACING)) > 0.1
        env = env[keep]
        spec_arr = to_spec_arr(env[:, 0], env[:, 1])
        assert envelope_is_halogen(spec_arr), \
            "Br2 envelope should be detected even with M+1 absent"

