"""
Tests for isotope envelope functionality.
"""

import numpy as np
from molmass import Formula
from find_mfs.isotopes.envelope import (
    get_isotope_envelope,
    match_isotope_envelope,
)

NOVOBIOCIN = "C31H36N2O11"
VANCOMYCIN = "C66H75Cl2N9O24"

class TestIsotopeEnvelope:
    def test_get_isotope_envelope(self):
        """
        Test basic isotope envelope calculation
        """
        novobiocin = Formula(NOVOBIOCIN)
        envelope = get_isotope_envelope(
            novobiocin,
            mz_tolerance=0.05,
            threshold=0.001,
        )

        # Should return an array
        assert isinstance(envelope, np.ndarray)

        # Array should be 2D
        assert envelope.ndim == 2, "Envelope is misformed"

        # Should have multiple peaks
        assert len(envelope) > 1, "Should have isotopologue peaks"

    def test_isotope_envelope_scaling(self):
        """
        Test that isotope envelopes are properly scaled
        """
        envelope = get_isotope_envelope(
            Formula(NOVOBIOCIN),
            mz_tolerance=0.05,
            threshold=0.001,
        )

        # Monoisotopic peak should be 1.0
        assert envelope[:, 1].max() == 1.0

        # All other peaks should be higher than 0.0
        assert envelope[:, 1].min() > 0.0

    def test_match_isotope_envelope(self):
        """
        Test isotope envelope matching with RMSE scoring
        """
        novobiocin_envelope = get_isotope_envelope(
            formula=Formula(NOVOBIOCIN),
            mz_tolerance=0.05,
            threshold=0.001,
        )

        # Should match itself perfectly (RMSE ~ 0)
        match_result = match_isotope_envelope(
            formula=Formula(NOVOBIOCIN),
            observed_envelope=novobiocin_envelope,
            mz_match_tolerance=0.01,
        )
        assert match_result.predicted_envelope.ndim == 2
        assert match_result.predicted_envelope.shape[1] == 2
        assert match_result.predicted_envelope.shape[0] > 0
        assert match_result.match_fraction == 1.0
        assert match_result.intensity_rmse < 0.01

        # Should not match well with different formula (higher RMSE)
        cross_result = match_isotope_envelope(
            formula=Formula(VANCOMYCIN),
            observed_envelope=novobiocin_envelope,
            mz_match_tolerance=0.01,
        )
        assert cross_result.match_fraction < 1.0
        assert cross_result.intensity_rmse > match_result.intensity_rmse

    def test_get_isotope_envelope_charged_formula_mz_alignment(self):
        # Ensure charge handling matches molmass conventions and that
        # multi-charge formulas don't rely on brittle string stripping.
        for formula_str in [
            "C6H11O6-",
            "[C6H11O6]2+",
        ]:
            formula = Formula(formula_str)
            envelope = get_isotope_envelope(
                formula,
                mz_tolerance=0.05,
                threshold=0.001,
            )
            assert envelope.shape[0] > 0
            target_mz = (
                formula.monoisotopic_mass
                if formula.charge == 0 else
                formula.monoisotopic_mass / abs(formula.charge)
            )
            closest = float(
                np.min(np.abs(envelope[:, 0] - target_mz))
            )
            assert closest < 2e-4
