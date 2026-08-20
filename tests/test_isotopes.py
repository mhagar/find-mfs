"""
Tests for isotope envelope functionality.
"""

import numpy as np
from molmass import Formula
from find_mfs.isotopes.envelope import get_isotope_envelope

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

    def test_isotope_loglik_self_match_is_perfect(self):
        """
        Feeding a formula's own simulated envelope back as the observed data
        should yield an (essentially) perfect isotope log-likelihood (~0).
        """
        from find_mfs.scoring.likelihoods import isotope_loglik

        novobiocin = Formula(NOVOBIOCIN)
        envelope = get_isotope_envelope(
            formula=novobiocin,
            mz_tolerance=0.05,
            threshold=0.001,
        )
        self_ll = isotope_loglik(novobiocin, envelope, mz_match_da=0.01)
        assert self_ll is not None
        assert self_ll > -1e-6  # ~0, best possible

    def test_isotope_loglik_penalizes_wrong_intensities(self):
        """
        A formula scored against its own simulated envelope (self-match) should
        score higher (closer to 0) than a different, near-isobaric formula whose
        isotope pattern is genuinely different at the same nominal m/z positions.
        """
        from find_mfs.scoring.likelihoods import isotope_loglik

        # Same C/H skeleton, S vs O2 swapped (~0.018 Da apart) so predicted peaks
        # land in the same m/z neighborhood, but sulfur's ~4.2% 34S gives `good`
        # a much stronger M+2 than the sulfur-free `bad` formula can reproduce.
        good = Formula("C10H12O4S")
        bad = Formula("C10H12O6")
        observed = get_isotope_envelope(good, mz_tolerance=0.05, threshold=0.001)

        good_ll = isotope_loglik(good, observed, mz_match_da=0.025)
        bad_ll = isotope_loglik(bad, observed, mz_match_da=0.025)

        assert good_ll is not None and bad_ll is not None
        assert good_ll > bad_ll

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
