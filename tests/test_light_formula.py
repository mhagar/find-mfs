"""Tests for LightFormula equivalence with molmass.Formula."""

import pytest
from molmass import Formula
from molmass.elements import ELEMENTS

from find_mfs.core.light_formula import LightFormula
from find_mfs import FormulaFinder


# ---------------------------------------------------------------------------
# Helper to build a LightFormula from a molmass Formula (for test comparisons)
# ---------------------------------------------------------------------------

def _light_from_formula(f: Formula) -> LightFormula:
    """Construct a LightFormula with the same data as a molmass.Formula."""
    elements = {}
    for sym, item in f.composition().items():
        if sym == '' or sym == 'e-':
            continue
        if item.count > 0:
            elements[sym] = item.count
    return LightFormula(
        elements=elements,
        charge=f.charge,
        monoisotopic_mass=f.monoisotopic_mass,
    )


# ---------------------------------------------------------------------------
# Formula string (Hill notation)
# ---------------------------------------------------------------------------

class TestFormulaString:
    @pytest.mark.parametrize("elements,charge,expected", [
        ({'C': 6, 'H': 12, 'O': 6}, 0, 'C6H12O6'),
        ({'C': 1, 'H': 4}, 0, 'CH4'),
        ({'H': 2, 'O': 1}, 0, 'H2O'),
        ({'N': 1, 'H': 3}, 0, 'H3N'),       # No C -> Hill: alphabetical
        ({'C': 5, 'H': 10, 'N': 2, 'O': 3, 'P': 1, 'S': 2}, 0, 'C5H10N2O3PS2'),
        ({'C': 6, 'H': 12, 'O': 6}, 1, '[C6H12O6]+'),
        ({'C': 6, 'H': 12, 'O': 6}, -1, '[C6H12O6]-'),
        ({'C': 6, 'H': 12, 'O': 6}, 2, '[C6H12O6]2+'),
        ({'C': 6, 'H': 12, 'O': 6}, -2, '[C6H12O6]2-'),
    ])
    def test_formula_string_hill_notation(self, elements, charge, expected):
        lf = LightFormula(elements=elements, charge=charge)
        assert lf.formula == expected

    def test_formula_string_matches_molmass(self):
        """Cross-check against molmass for several formulas."""
        test_cases = ['C6H12O6', 'CH4', 'C5H10N2O3PS2']
        for formula_str in test_cases:
            f = Formula(formula_str)
            lf = _light_from_formula(f)
            assert lf.formula == f.formula, f"Mismatch for {formula_str}"

    def test_formula_string_charged_matches_molmass(self):
        for formula_str in ['C6H12O6+', 'C6H12O6-', 'C6H12O6++']:
            f = Formula(formula_str)
            lf = _light_from_formula(f)
            assert lf.formula == f.formula, f"Mismatch for {formula_str}"


# ---------------------------------------------------------------------------
# Monoisotopic mass
# ---------------------------------------------------------------------------

class TestMonoisotopicMass:
    @pytest.mark.parametrize("formula_str", [
        'C6H12O6',
        'CH4',
        'C5H10N2O3PS2',
        'C20H30N5O10PS',
    ])
    def test_monoisotopic_mass_matches(self, formula_str):
        f = Formula(formula_str)
        lf = _light_from_formula(f)
        assert abs(lf.monoisotopic_mass - f.monoisotopic_mass) < 1e-10


# ---------------------------------------------------------------------------
# Charge
# ---------------------------------------------------------------------------

class TestCharge:
    @pytest.mark.parametrize("charge", [0, 1, -1, 2, -2])
    def test_charge_property(self, charge):
        lf = LightFormula(elements={'C': 6, 'H': 12, 'O': 6}, charge=charge)
        assert lf.charge == charge


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

class TestComposition:
    @pytest.mark.parametrize("formula_str", [
        'C6H12O6',
        'CH4',
        'C5H10N2O3PS2',
    ])
    def test_composition_symbols_and_counts_match(self, formula_str):
        f = Formula(formula_str)
        lf = _light_from_formula(f)

        f_comp = f.composition()
        lf_comp = lf.composition()

        # Same element keys (excluding '' total row which molmass may have)
        f_keys = {k for k in f_comp if k != ''}
        lf_keys = set(lf_comp.keys())
        assert f_keys == lf_keys, f"Key mismatch for {formula_str}"

        for sym in f_keys:
            if sym == 'e-':
                continue
            assert lf_comp[sym].symbol == f_comp[sym].symbol
            assert lf_comp[sym].count == f_comp[sym].count

    @pytest.mark.parametrize("formula_str", [
        'C6H12O6',
        'C5H10N2O3PS2',
    ])
    def test_composition_mass_fraction(self, formula_str):
        f = Formula(formula_str)
        lf = _light_from_formula(f)

        f_comp = f.composition()
        lf_comp = lf.composition()

        for sym in lf_comp:
            if sym == 'e-':
                continue
            assert abs(lf_comp[sym].mass - f_comp[sym].mass) < 1e-6, \
                f"Mass mismatch for {sym} in {formula_str}"
            assert abs(lf_comp[sym].fraction - f_comp[sym].fraction) < 1e-6, \
                f"Fraction mismatch for {sym} in {formula_str}"

    def test_composition_charged_has_electron_entry(self):
        lf = LightFormula(elements={'C': 6, 'H': 12, 'O': 6}, charge=1)
        comp = lf.composition()
        assert 'e-' in comp
        assert comp['e-'].count == -1

    def test_composition_values_iterable(self):
        """Verify .composition().values() works (used by filtering.py)."""
        lf = LightFormula(elements={'C': 6, 'H': 12, 'O': 6}, charge=0)
        symbols = [item.symbol for item in lf.composition().values()]
        assert 'C' in symbols
        assert 'H' in symbols
        assert 'O' in symbols


# ---------------------------------------------------------------------------
# Addition
# ---------------------------------------------------------------------------

class TestAddition:
    def test_add_light_formulas(self):
        lf1 = LightFormula({'C': 6, 'H': 12, 'O': 6}, charge=1, monoisotopic_mass=180.0)
        lf2 = LightFormula({'Na': 1}, charge=0, monoisotopic_mass=23.0)
        result = lf1 + lf2
        assert result._elements == {'C': 6, 'H': 12, 'O': 6, 'Na': 1}
        assert result.charge == 1
        assert abs(result.monoisotopic_mass - 203.0) < 1e-10

    def test_add_light_formula_and_formula(self):
        """The adduct path: LightFormula + molmass.Formula."""
        lf = LightFormula({'C': 6, 'H': 12, 'O': 6}, charge=1, monoisotopic_mass=180.063)
        adduct = Formula('Na')
        result = lf + adduct
        assert result._elements['Na'] == 1
        assert result._elements['C'] == 6
        assert result.charge == 1  # Na is neutral
        assert abs(result.monoisotopic_mass - (180.063 + adduct.monoisotopic_mass)) < 1e-10

    def test_add_preserves_light_formula_type(self):
        lf1 = LightFormula({'C': 1}, charge=0, monoisotopic_mass=12.0)
        lf2 = LightFormula({'H': 4}, charge=0, monoisotopic_mass=4.0)
        result = lf1 + lf2
        assert isinstance(result, LightFormula)

    def test_add_with_molmass_returns_light(self):
        lf = LightFormula({'C': 1}, charge=0, monoisotopic_mass=12.0)
        f = Formula('H4')
        result = lf + f
        assert isinstance(result, LightFormula)


# ---------------------------------------------------------------------------
# to_formula escape hatch
# ---------------------------------------------------------------------------

class TestToFormula:
    def test_to_formula_roundtrip(self):
        lf = LightFormula(
            elements={'C': 6, 'H': 12, 'O': 6},
            charge=0,
            monoisotopic_mass=180.063388,
        )
        real = lf.to_formula()
        assert isinstance(real, Formula)
        assert real.formula == lf.formula
        assert abs(real.monoisotopic_mass - lf.monoisotopic_mass) < 0.01

    def test_to_formula_charged(self):
        lf = LightFormula(
            elements={'C': 6, 'H': 12, 'O': 6},
            charge=1,
            monoisotopic_mass=180.063,
        )
        real = lf.to_formula()
        assert real.charge == 1


# ---------------------------------------------------------------------------
# String representations
# ---------------------------------------------------------------------------

class TestStrRepr:
    def test_str(self):
        lf = LightFormula({'C': 6, 'H': 12, 'O': 6}, charge=0)
        assert str(lf) == 'C6H12O6'

    def test_repr(self):
        lf = LightFormula({'C': 6, 'H': 12, 'O': 6}, charge=0)
        assert repr(lf) == "LightFormula('C6H12O6')"


# ---------------------------------------------------------------------------
# End-to-end equivalence with FormulaFinder
# ---------------------------------------------------------------------------

class TestEndToEnd:
    @pytest.fixture(scope='class')
    def finder(self):
        return FormulaFinder('CHNOPS')

    @pytest.mark.parametrize("mass,charge,adduct,error_ppm,check_octet,filter_rdbe", [
        # Basic neutral search
        (180.063388, 0, None, 5.0, False, None),
        # Charged search
        (181.071064, 1, None, 5.0, False, None),
        # With octet + RDBE filtering
        (180.063388, 0, None, 5.0, True, (0, 20)),
        # With adduct
        (203.053, 1, 'Na', 10.0, False, None),
        # Negative mode
        (179.056, -1, '-H', 10.0, False, None),
        # Larger mass (more candidates)
        (500.0, 1, None, 10.0, True, (0, 20)),
    ])
    def test_end_to_end_equivalence(
        self, finder, mass, charge, adduct, error_ppm, check_octet, filter_rdbe
    ):
        results = finder.find_formulae(
            mass=mass,
            charge=charge,
            adduct=adduct,
            error_ppm=error_ppm,
            check_octet=check_octet,
            filter_rdbe=filter_rdbe,
            max_counts='C50H80N10O20P5S3',
        )

        for candidate in results:
            lf = candidate.formula
            # Verify it's a LightFormula
            assert isinstance(lf, LightFormula), \
                f"Expected LightFormula, got {type(lf)}"

            # Build the equivalent molmass.Formula for comparison
            real = Formula(lf.formula)

            # Formula string should be valid and parseable
            assert real.formula == lf.formula, \
                f"Formula string mismatch: {lf.formula} vs {real.formula}"

            # Monoisotopic mass should match closely
            assert abs(lf.monoisotopic_mass - real.monoisotopic_mass) < 1e-6, \
                f"Mass mismatch for {lf.formula}: {lf.monoisotopic_mass} vs {real.monoisotopic_mass}"

            # Charge should match
            assert lf.charge == real.charge

            # Composition symbols and counts should match
            lf_comp = lf.composition()
            real_comp = real.composition()
            for sym in lf_comp:
                if sym == 'e-':
                    continue
                assert sym in real_comp, \
                    f"Symbol {sym} missing from real composition"
                assert lf_comp[sym].count == real_comp[sym].count, \
                    f"Count mismatch for {sym}: {lf_comp[sym].count} vs {real_comp[sym].count}"

    def test_novobiocin(self, finder):
        """Verify the classic novobiocin test still works."""
        results = finder.find_formulae(
            mass=189.09133,
            charge=1,
            error_ppm=5.0,
        )
        assert len(results) > 0
        # All candidates should be LightFormula instances
        for c in results:
            assert isinstance(c.formula, LightFormula)
