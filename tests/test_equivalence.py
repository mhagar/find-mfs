"""
Tests for match_isotope_envelope correctness across diverse formulae,
charge states, and molecular compositions.

Verifies that RMSE and match results are consistent and sensible.
"""

import numpy as np
import pytest
from molmass import Formula

from find_mfs.isotopes.envelope import (
    get_isotope_envelope,
    match_isotope_envelope,
)


# Test formulae covering diverse chemical space
TEST_FORMULAE = [
    # (formula_str, description)
    ("C6H12O6", "glucose - small organic"),
    ("C9H8O4", "aspirin - small aromatic"),
    ("C31H36N2O11", "novobiocin - mid-size natural product"),
    ("C66H75Cl2N9O24", "vancomycin - large halogenated"),
    ("C10H16N5O13P3", "ATP - phosphorus-rich"),
    ("C3H7NO2S", "cysteine - sulfur-containing"),
    ("C10H15N3O6S2", "cystine analog - high sulfur"),
    ("C2H3Cl3", "trichloroethane - halogenated"),
    ("C20H25N3O", "medium nitrogen-rich"),
    ("C5H5N5", "adenine - high N/C ratio"),
]


def _make_observed_envelope(formula_str: str, charge: int = 0):
    """Generate an observed envelope from formula."""
    if charge != 0:
        sign = "+" if charge > 0 else "-"
        n = abs(charge)
        charged_str = f"[{formula_str}]{sign}" if n == 1 else f"[{formula_str}]{sign}{n}"
        f = Formula(charged_str)
    else:
        f = Formula(formula_str)

    envelope = get_isotope_envelope(f, mz_tolerance=0.05, threshold=0.001)
    return envelope, f


class TestSelfMatch:
    """Self-match tests: scoring a formula against its own envelope."""

    @pytest.mark.parametrize("formula_str,desc", TEST_FORMULAE)
    def test_self_match_low_rmse(self, formula_str, desc):
        """Self-matching should produce near-zero RMSE."""
        envelope, formula_obj = _make_observed_envelope(formula_str)

        result = match_isotope_envelope(
            formula=formula_obj,
            observed_envelope=envelope,
            mz_match_tolerance=0.01,
            simulated_envelope_mz_tolerance=0.05,
            simulated_envelope_intsy_threshold=0.001,
        )

        assert result.intensity_rmse < 0.02, (
            f"{desc}: self-match RMSE too high: {result.intensity_rmse:.6f}"
        )

    @pytest.mark.parametrize("formula_str,desc", TEST_FORMULAE)
    def test_self_match_all_peaks(self, formula_str, desc):
        """Self-matching should match all observed peaks."""
        envelope, formula_obj = _make_observed_envelope(formula_str)

        result = match_isotope_envelope(
            formula=formula_obj,
            observed_envelope=envelope,
            mz_match_tolerance=0.01,
            simulated_envelope_mz_tolerance=0.05,
            simulated_envelope_intsy_threshold=0.001,
        )

        assert result.num_peaks_matched == result.num_peaks_total, (
            f"{desc}: not all peaks matched: "
            f"{result.num_peaks_matched}/{result.num_peaks_total}"
        )

    @pytest.mark.parametrize("formula_str,desc", TEST_FORMULAE)
    def test_peak_matches_array(self, formula_str, desc):
        """peak_matches array should have correct dtype, length, and values."""
        envelope, formula_obj = _make_observed_envelope(formula_str)

        result = match_isotope_envelope(
            formula=formula_obj,
            observed_envelope=envelope,
            mz_match_tolerance=0.01,
        )

        assert result.peak_matches.dtype == bool
        assert len(result.peak_matches) == envelope.shape[0]
        assert np.sum(result.peak_matches) == result.num_peaks_matched


class TestChargedSpecies:
    """Tests for charged species."""

    @pytest.mark.parametrize("charge", [1, -1])
    def test_glucose_charged(self, charge):
        """Charged glucose should still self-match well."""
        envelope, formula_obj = _make_observed_envelope("C6H12O6", charge=charge)

        result = match_isotope_envelope(
            formula=formula_obj,
            observed_envelope=envelope,
            mz_match_tolerance=0.01,
        )

        assert result.num_peaks_matched == result.num_peaks_total
        assert result.intensity_rmse < 0.02

    @pytest.mark.parametrize("charge", [1, -1])
    def test_vancomycin_charged(self, charge):
        """Large molecule charged should still self-match well."""
        envelope, formula_obj = _make_observed_envelope("C66H75Cl2N9O24", charge=charge)

        result = match_isotope_envelope(
            formula=formula_obj,
            observed_envelope=envelope,
            mz_match_tolerance=0.01,
        )

        assert result.num_peaks_matched == result.num_peaks_total
        assert result.intensity_rmse < 0.02


class TestCrossFormulaMismatch:
    """Verify that mismatched formulae produce elevated RMSE."""

    def test_glucose_vs_aspirin(self):
        """Cross-formula matching should produce higher RMSE than self-match."""
        glucose_env, _ = _make_observed_envelope("C6H12O6")
        aspirin_formula = Formula("C9H8O4")

        result = match_isotope_envelope(
            formula=aspirin_formula,
            observed_envelope=glucose_env,
            mz_match_tolerance=0.5,
        )

        assert result.intensity_rmse > 0.01


class TestPerPeakMatchDetail:
    """Verify that peak_matches reports real per-peak data, not placeholders."""

    def test_bogus_peak_not_matched(self):
        """A fabricated peak far from any predicted peak should not match."""
        envelope = np.array([
            [180.0634, 1.00],   # M+0
            [181.0667, 0.11],   # M+1
            [185.0000, 0.05],   # Bogus — should NOT match
        ])
        formula = Formula("C6H12O6")

        result = match_isotope_envelope(
            formula=formula,
            observed_envelope=envelope,
            mz_match_tolerance=0.01,
        )

        assert result.peak_matches[0] == True,  "M+0 should match"
        assert result.peak_matches[1] == True,  "M+1 should match"
        assert result.peak_matches[2] == False, "Bogus peak should NOT match"
        assert result.num_peaks_matched == 2
