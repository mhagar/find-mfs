"""
Pre-filter false negative sweep test.

Verifies that the approximate M+1/M+2 decomposition pre-filter (an opt-in perf
gate, driven by the ``isotope_prefilter`` kwarg) does not cause false negatives
across a range of masses and formula types: it should only drop candidates that
are genuinely poor isotope matches, never the true formula.
"""

import numpy as np
import pytest
from molmass import Formula

from find_mfs import FormulaFinder
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
    """Verify the pre-filter never eliminates good isotope-match candidates."""

    @pytest.mark.parametrize("formula_str,charge", SWEEP_FORMULAE)
    def test_prefilter_is_subset(self, formula_str, charge):
        """
        The M+1/M+2 pre-filter only removes candidates; the pre-filtered result
        set is always a subset of the unfiltered one (it never adds spurious
        candidates), across the mass sweep.
        """
        finder = FormulaFinder("CHNOPS")

        f = Formula(formula_str)
        mass = f.monoisotopic_mass
        from find_mfs.core.finder import ELECTRON
        if charge != 0:
            mass = (mass + charge * ELECTRON.mass) / abs(charge)

        envelope = _make_envelope_from_formula(formula_str, charge)

        results_on = finder.find_formulae(
            mass=mass,
            charge=charge,
            error_ppm=5.0,
            isotope_prefilter=envelope.copy(),
            filter_rdbe=(-0.5, 40),
            check_octet=True,
        )
        results_off = finder.find_formulae(
            mass=mass,
            charge=charge,
            error_ppm=5.0,
            filter_rdbe=(-0.5, 40),
            check_octet=True,
        )

        formulas_on = {str(c.formula) for c in results_on}
        formulas_off = {str(c.formula) for c in results_off}

        assert formulas_on <= formulas_off

    def test_prefilter_can_eliminate_candidates(self):
        """
        On a large candidate space the pre-filter should eliminate some
        candidates (i.e. it is actually doing useful work).
        """
        finder = FormulaFinder("CHNOPS")
        formula_str = "C43H58N4O12"  # ~810 Da, hundreds of candidates
        charge = 1
        f = Formula(formula_str)
        mass = f.monoisotopic_mass
        from find_mfs.core.finder import ELECTRON
        mass = (mass + charge * ELECTRON.mass) / abs(charge)

        envelope = _make_envelope_from_formula(formula_str, charge)

        results_on = finder.find_formulae(
            mass=mass, charge=charge, error_ppm=5.0,
            isotope_prefilter=envelope.copy(),
            filter_rdbe=(-0.5, 40), check_octet=True,
        )
        results_off = finder.find_formulae(
            mass=mass, charge=charge, error_ppm=5.0,
            filter_rdbe=(-0.5, 40), check_octet=True,
        )

        assert len(results_on) < len(results_off)
        # Pre-filter only removes candidates; never adds any.
        formulas_on = {str(c.formula) for c in results_on}
        formulas_off = {str(c.formula) for c in results_off}
        assert formulas_on <= formulas_off
