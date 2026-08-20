"""
Tests for halogen detection
"""
from molmass import Formula
from find_mfs.spectra.halogen import envelope_is_halogen
from find_mfs.isotopes.envelope import get_isotope_envelope
from find_mfs.spectra.envelopes import to_spec_arr

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

