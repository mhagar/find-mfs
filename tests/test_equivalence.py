"""
Wide-range equivalence tests: old match_isotope_envelope vs new match_isotope_envelope_fast.

Verifies that RMSE and match results are consistent across diverse formulae,
charge states, and molecular compositions.
"""

import numpy as np
import pytest
from molmass import Formula

from find_mfs.isotopes.envelope import (
    get_isotope_envelope,
    match_isotope_envelope,
    match_isotope_envelope_fast,
)


# Test formulae covering diverse chemical space
EQUIVALENCE_FORMULAE = [
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
    """Generate an observed envelope from formula via old path."""
    if charge != 0:
        # Use molmass charge notation: [formula]+n or [formula]-n
        sign = "+" if charge > 0 else "-"
        n = abs(charge)
        charged_str = f"[{formula_str}]{sign}" if n == 1 else f"[{formula_str}]{sign}{n}"
        f = Formula(charged_str)
    else:
        f = Formula(formula_str)

    envelope = get_isotope_envelope(f, mz_tolerance=0.05, threshold=0.001)
    return envelope, f


def _get_symbols_counts(formula_str: str):
    """Parse formula into symbols and counts lists."""
    f = Formula(formula_str)
    composition = f.composition()
    symbols = []
    counts = []
    for element, item in sorted(composition.items()):
        symbols.append(element)
        counts.append(item.count)
    return symbols, counts


class TestEquivalenceNeutral:
    """Equivalence tests for neutral (charge=0) formulae."""

    @pytest.mark.parametrize("formula_str,desc", EQUIVALENCE_FORMULAE)
    def test_rmse_equivalence(self, formula_str, desc):
        """RMSE from old and new paths should match within tolerance."""
        envelope, formula_obj = _make_observed_envelope(formula_str)
        symbols, counts = _get_symbols_counts(formula_str)

        # Old path
        old_result = match_isotope_envelope(
            formula=formula_obj,
            observed_envelope=envelope,
            mz_match_tolerance=0.01,
            simulated_envelope_mz_tolerance=0.05,
            simulated_envelope_intsy_threshold=0.001,
        )

        # New path
        new_result = match_isotope_envelope_fast(
            symbols=symbols,
            counts=counts,
            charge=0,
            observed_envelope=envelope,
            mz_match_tolerance=0.01,
            simulated_mz_tolerance=0.05,
            simulated_intensity_threshold=0.001,
        )

        # Compare RMSE - allow for float32 vs float64 differences
        if old_result.intensity_rmse == 0.0:
            assert new_result.intensity_rmse < 1e-4, (
                f"{desc}: old RMSE=0 but new RMSE={new_result.intensity_rmse}"
            )
        else:
            rel_diff = abs(old_result.intensity_rmse - new_result.intensity_rmse) / max(
                old_result.intensity_rmse, 1e-10
            )
            assert rel_diff < 0.05, (  # 5% relative tolerance for float32→64
                f"{desc}: RMSE mismatch: old={old_result.intensity_rmse:.6f}, "
                f"new={new_result.intensity_rmse:.6f}, rel_diff={rel_diff:.4f}"
            )

    @pytest.mark.parametrize("formula_str,desc", EQUIVALENCE_FORMULAE)
    def test_match_count_equivalence(self, formula_str, desc):
        """Number of matched peaks should be identical."""
        envelope, formula_obj = _make_observed_envelope(formula_str)
        symbols, counts = _get_symbols_counts(formula_str)

        old_result = match_isotope_envelope(
            formula=formula_obj,
            observed_envelope=envelope,
            mz_match_tolerance=0.01,
            simulated_envelope_mz_tolerance=0.05,
            simulated_envelope_intsy_threshold=0.001,
        )

        new_result = match_isotope_envelope_fast(
            symbols=symbols,
            counts=counts,
            charge=0,
            observed_envelope=envelope,
            mz_match_tolerance=0.01,
            simulated_mz_tolerance=0.05,
            simulated_intensity_threshold=0.001,
        )

        assert old_result.num_peaks_matched == new_result.num_peaks_matched, (
            f"{desc}: match count mismatch: "
            f"old={old_result.num_peaks_matched}, new={new_result.num_peaks_matched}"
        )


class TestEquivalenceCharged:
    """Equivalence tests for charged species."""

    @pytest.mark.parametrize("charge", [1, -1])
    def test_glucose_charged(self, charge):
        """Charged glucose should give equivalent results (charge ±1)."""
        formula_str = "C6H12O6"
        envelope, formula_obj = _make_observed_envelope(formula_str, charge=charge)
        symbols, counts = _get_symbols_counts(formula_str)

        old_result = match_isotope_envelope(
            formula=formula_obj,
            observed_envelope=envelope,
            mz_match_tolerance=0.01,
            simulated_envelope_mz_tolerance=0.05,
            simulated_envelope_intsy_threshold=0.001,
        )

        new_result = match_isotope_envelope_fast(
            symbols=symbols,
            counts=counts,
            charge=charge,
            observed_envelope=envelope,
            mz_match_tolerance=0.01,
            simulated_mz_tolerance=0.05,
            simulated_intensity_threshold=0.001,
        )

        assert old_result.num_peaks_matched == new_result.num_peaks_matched
        if old_result.intensity_rmse > 0:
            rel_diff = abs(old_result.intensity_rmse - new_result.intensity_rmse) / old_result.intensity_rmse
            assert rel_diff < 0.05

    @pytest.mark.parametrize("charge", [1, -1])
    def test_vancomycin_charged(self, charge):
        """Large molecule charged should give equivalent results."""
        formula_str = "C66H75Cl2N9O24"
        envelope, formula_obj = _make_observed_envelope(formula_str, charge=charge)
        symbols, counts = _get_symbols_counts(formula_str)

        old_result = match_isotope_envelope(
            formula=formula_obj,
            observed_envelope=envelope,
            mz_match_tolerance=0.01,
            simulated_envelope_mz_tolerance=0.05,
            simulated_envelope_intsy_threshold=0.001,
        )

        new_result = match_isotope_envelope_fast(
            symbols=symbols,
            counts=counts,
            charge=charge,
            observed_envelope=envelope,
            mz_match_tolerance=0.01,
            simulated_mz_tolerance=0.05,
            simulated_intensity_threshold=0.001,
        )

        assert old_result.num_peaks_matched == new_result.num_peaks_matched
        if old_result.intensity_rmse > 0:
            rel_diff = abs(old_result.intensity_rmse - new_result.intensity_rmse) / old_result.intensity_rmse
            assert rel_diff < 0.05


class TestCrossFormulaMismatch:
    """Verify that mismatched formulae produce similar RMSE in both paths."""

    def test_glucose_vs_aspirin(self):
        """Cross-formula matching should give similar RMSE in both paths."""
        # Use glucose envelope, score against aspirin
        glucose_env, _ = _make_observed_envelope("C6H12O6")
        aspirin_formula = Formula("C9H8O4")

        old_result = match_isotope_envelope(
            formula=aspirin_formula,
            observed_envelope=glucose_env,
            mz_match_tolerance=0.5,
            simulated_envelope_mz_tolerance=0.05,
            simulated_envelope_intsy_threshold=0.001,
        )

        symbols, counts = _get_symbols_counts("C9H8O4")
        new_result = match_isotope_envelope_fast(
            symbols=symbols,
            counts=counts,
            charge=0,
            observed_envelope=glucose_env,
            mz_match_tolerance=0.5,
            simulated_mz_tolerance=0.05,
            simulated_intensity_threshold=0.001,
        )

        # Both should detect poor match
        assert old_result.intensity_rmse > 0.01
        assert new_result.intensity_rmse > 0.01

        # RMSE should be similar
        if old_result.intensity_rmse > 0:
            rel_diff = abs(old_result.intensity_rmse - new_result.intensity_rmse) / old_result.intensity_rmse
            assert rel_diff < 0.1  # 10% tolerance for cross-formula


class TestDtypeDifference:
    """Document and verify the float32 vs float64 difference (Issue 2)."""

    def test_float_precision_impact(self):
        """
        Quantify the impact of float32 (old path) vs float64 (new path).

        The old path uses np.float32 in get_isotope_envelope, which loses
        precision during combine_unresolved_isotopologues. The new path
        uses float64 throughout. This test documents the magnitude of
        the difference.
        """
        formula_str = "C66H75Cl2N9O24"  # Large molecule, many isotopologues
        envelope, formula_obj = _make_observed_envelope(formula_str)
        symbols, counts = _get_symbols_counts(formula_str)

        old_result = match_isotope_envelope(
            formula=formula_obj,
            observed_envelope=envelope,
            mz_match_tolerance=0.01,
            simulated_envelope_mz_tolerance=0.05,
            simulated_envelope_intsy_threshold=0.001,
        )

        new_result = match_isotope_envelope_fast(
            symbols=symbols,
            counts=counts,
            charge=0,
            observed_envelope=envelope,
            mz_match_tolerance=0.01,
            simulated_mz_tolerance=0.05,
            simulated_intensity_threshold=0.001,
        )

        abs_diff = abs(old_result.intensity_rmse - new_result.intensity_rmse)
        print(f"\n  Float precision impact on vancomycin:")
        print(f"  Old RMSE (float32 intermediate): {old_result.intensity_rmse:.8f}")
        print(f"  New RMSE (float64 throughout):   {new_result.intensity_rmse:.8f}")
        print(f"  Absolute difference:             {abs_diff:.8f}")

        # The difference should be small enough not to affect filtering
        assert abs_diff < 0.01, (
            f"Float precision difference too large: {abs_diff}"
        )
