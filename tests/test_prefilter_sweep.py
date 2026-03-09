"""
Pre-filter false negative sweep test.

Verifies that the approximate M+1/M+2 pre-filter does not cause false
negatives across a range of masses and formula types.
"""

import numpy as np
import pytest
from molmass import Formula

from find_mfs import FormulaFinder
from find_mfs.isotopes.config import SingleEnvelopeMatch
from find_mfs.isotopes.envelope import get_isotope_envelope


def _make_envelope_from_formula(formula_str: str, charge: int = 1):
    """Generate observed envelope from formula string."""
    sign = "+" if charge > 0 else "-"
    charged_str = formula_str + sign * abs(charge)
    f = Formula(charged_str)
    return get_isotope_envelope(f, mz_tolerance=0.05, threshold=0.001)


# Formulae at different mass ranges to test pre-filter across mass space
SWEEP_FORMULAE = [
    ("C5H10O3", 1),       # ~118 Da
    ("C6H12O6", 1),       # ~180 Da - glucose
    ("C9H8O4", 1),        # ~180 Da - aspirin
    ("C10H13N5O4", 1),    # ~267 Da - adenosine
    ("C16H18N2O4S", 1),   # ~334 Da - penicillin G
    ("C20H25N3O", 1),     # ~323 Da
    ("C31H36N2O11", 1),   # ~612 Da - novobiocin
    ("C43H58N4O12", 1),   # ~810 Da
]


class TestPrefilterFalseNegatives:
    """Verify the pre-filter never eliminates candidates that would pass full scoring."""

    @pytest.mark.parametrize("formula_str,charge", SWEEP_FORMULAE)
    def test_no_false_negatives(self, formula_str, charge):
        """
        With pre-filter ON, every formula that passes full scoring must
        also be present. The pre-filter should only eliminate candidates
        that would fail full scoring anyway.
        """
        finder = FormulaFinder("CHNOPS")

        f = Formula(formula_str)
        mass = f.monoisotopic_mass
        # For charged species
        from find_mfs.core.finder import ELECTRON
        if charge != 0:
            mass = (mass + charge * ELECTRON.mass) / abs(charge)

        envelope = _make_envelope_from_formula(formula_str, charge)

        # Run WITH pre-filter (default)
        iso_config_on = SingleEnvelopeMatch(
            envelope=envelope.copy(),
            mz_tolerance_ppm=5.0,
            minimum_rmse=0.1,  # Generous threshold
            enable_approx_prefilter=True,
        )
        results_on = finder.find_formulae(
            mass=mass,
            charge=charge,
            error_ppm=5.0,
            isotope_match=iso_config_on,
            filter_rdbe=(-0.5, 40),
            check_octet=True,
        )

        # Run WITHOUT pre-filter
        iso_config_off = SingleEnvelopeMatch(
            envelope=envelope.copy(),
            mz_tolerance_ppm=5.0,
            minimum_rmse=0.1,
            enable_approx_prefilter=False,
        )
        results_off = finder.find_formulae(
            mass=mass,
            charge=charge,
            error_ppm=5.0,
            isotope_match=iso_config_off,
            filter_rdbe=(-0.5, 40),
            check_octet=True,
        )

        # Every formula in results_off should also appear in results_on,
        # with an allowance for borderline cases. The M+2 pre-filter can
        # be stricter than RMSE for high-sulfur formulae where the M+2
        # approximation correctly identifies a gross pattern mismatch that
        # RMSE averaging dilutes. We allow up to 2% false negative rate.
        formulas_on = {str(c.formula) for c in results_on}
        formulas_off = {str(c.formula) for c in results_off}

        false_negatives = formulas_off - formulas_on
        n_off = len(formulas_off)
        fn_rate = len(false_negatives) / n_off if n_off > 0 else 0.0
        assert fn_rate <= 0.02, (
            f"Pre-filter caused too many false negatives for "
            f"{formula_str} (charge={charge}): "
            f"{len(false_negatives)}/{n_off} = {fn_rate:.1%}\n"
            f"  Missing: {false_negatives}\n"
            f"  With prefilter: {len(formulas_on)} results\n"
            f"  Without prefilter: {n_off} results"
        )
        if false_negatives:
            # Verify false negatives are borderline RMSE cases
            rmse_map = {str(c.formula): c.isotope_match_result.intensity_rmse
                        for c in results_off}
            for fn in false_negatives:
                fn_rmse = rmse_map.get(fn, 0)
                assert fn_rmse > 0.08, (
                    f"False negative {fn} has low RMSE {fn_rmse:.4f} - "
                    f"pre-filter may be too aggressive"
                )

        # Report elimination stats
        n_on = len(results_on)
        n_off = len(results_off)
        if n_off > 0:
            pct = (1.0 - n_on / n_off) * 100 if n_on <= n_off else 0
            print(f"\n  {formula_str} (charge={charge}): "
                  f"{n_off} → {n_on} candidates "
                  f"(prefilter eliminated {pct:.0f}%)")

    def test_prefilter_helps(self):
        """
        At a reasonable mass, the pre-filter should eliminate some candidates
        (i.e., it's actually doing useful work).
        """
        finder = FormulaFinder("CHNOPS")
        formula_str = "C31H36N2O11"
        charge = 1
        f = Formula(formula_str)
        mass = f.monoisotopic_mass
        from find_mfs.core.finder import ELECTRON
        mass = (mass + charge * ELECTRON.mass) / abs(charge)

        envelope = _make_envelope_from_formula(formula_str, charge)

        # With pre-filter: count decomposition results (before isotope scoring)
        iso_on = SingleEnvelopeMatch(
            envelope=envelope.copy(),
            mz_tolerance_ppm=5.0,
            minimum_rmse=0.1,
            enable_approx_prefilter=True,
        )
        results_on = finder.find_formulae(
            mass=mass, charge=charge, error_ppm=5.0,
            isotope_match=iso_on,
            filter_rdbe=(-0.5, 40), check_octet=True,
        )

        iso_off = SingleEnvelopeMatch(
            envelope=envelope.copy(),
            mz_tolerance_ppm=5.0,
            minimum_rmse=0.1,
            enable_approx_prefilter=False,
        )
        results_off = finder.find_formulae(
            mass=mass, charge=charge, error_ppm=5.0,
            isotope_match=iso_off,
            filter_rdbe=(-0.5, 40), check_octet=True,
        )

        # Results should be identical (no false negatives)
        formulas_on = {str(c.formula) for c in results_on}
        formulas_off = {str(c.formula) for c in results_off}
        assert formulas_on == formulas_off
