"""
Tests for core mass decomposition functionality
"""

from molmass import Formula
from find_mfs.core.decomposer import MassDecomposer
from find_mfs.core.finder import FormulaFinder
from find_mfs.core.results import FormulaSearchResults
from find_mfs.utils import (
    passes_octet_rule, formula_match
)

class TestFormulaFinder:
    pass
    # TODO

class TestMassDecomposer:

    def setup_method(self):
        """
        Initialize mass decomposer
        """
        self.decomposer = MassDecomposer('CHNOPS')

    def test_initialization(self):
        """
        Test that decomposer initializes correctly
        """
        assert len(self.decomposer.elements) == 6
        assert set(self.decomposer.element_symbols) == set('CHNOPS')
        assert self.decomposer.ERT is not None

    def test_initialization_with_halogens(self):
        """
        Initialize mass decomposer with halogens
        """
        decomposer = MassDecomposer('CHNOPSBrClIF')
        assert len(decomposer.elements) == 10
        assert set(
            [str(x) for x in decomposer.element_symbols]
        ) == {
            'C', 'H', 'N', 'O', 'P', 'S',
            'Br', 'Cl', 'I', 'F'
        }
        assert decomposer.ERT is not None

    def test_simple_decomposition(self):
        """
        Test decomposition of H2O
        """
        water_mass = Formula('H2O').monoisotopic_mass

        results = self.decomposer.decompose(
            query_mass=water_mass,
            ppm_error=10.0,
            max_results=10
        )

        # Should find H2O
        assert len(results) > 0
        assert any(
            'H2O' in formula.formula for formula in results
        )

    def test_element_constraints(self):
        """
        Test that element count constraints work
        """
        results = self.decomposer.decompose(
            query_mass=100.0,
            ppm_error=10.0,
            min_counts={"C": 5},    # At least 5 carbons
            max_counts={            # Upper limits
                "C": 10,
                "H": 20,
                "O": 5,
            }
        )

        # Check that all results respect constraints
        for formula in results:
            composition = dict(formula.composition())
            assert composition.get('C', [0])[0] >= 5  # At least 5 carbons
            assert composition.get('C', [0])[0] <= 10 # At most 10 carbons
            assert composition.get('H', [0])[0] <= 20 # At most 10 hydrogens
            assert composition.get('O', [0])[0] <= 5  # At most 5 oxygens

    def test_rdbe_filtering(self):
        """
        Test RDBE-based filtering using FormulaFinder
        """
        test_formula = Formula('C6H6')   # Benzene
        mass = test_formula.monoisotopic_mass

        min_rdbe, max_rdbe = (2, 10)

        # Use FormulaFinder for filtering
        finder = FormulaFinder('CHNOPS')
        results = finder.find_formulae(
            mass=mass,
            error_ppm=100.0,
            max_results=50,
            filter_rdbe=(min_rdbe, max_rdbe)
        )

        assert isinstance(results, FormulaSearchResults)
        assert len(results) > 0
        result_rdbes: list[float] = [
            result.rdbe for result in results
        ]

        assert max(result_rdbes) <= max_rdbe, (
            f"RDBEs of results don't match range. "
            f"Max RDBE: {max(result_rdbes)}, requested max: {max_rdbe}"
        )

        assert min(result_rdbes) >= min_rdbe, (
            f"RDBEs of results don't match range. "
            f"Min RDBE: {min(result_rdbes)}, requested min: {min_rdbe}"
        )

        assert any(
            [
                formula_match(
                    test_formula,
                    result.formula,
                ) for result in results
            ]
        ), f"Test formula {test_formula} not found in results"

    def test_octet_rule_filtering(self):
        """
        Test octet-based filtering using FormulaFinder
        """
        test_formula = Formula('C6H7O+')  # Phenolium
        mass = test_formula.monoisotopic_mass

        # Use FormulaFinder for filtering
        finder = FormulaFinder('CHNOPS')
        results = finder.find_formulae(
            mass=mass,
            charge=1,
            error_ppm=100.0,
            max_results=50,
            check_octet=True,
        )

        assert isinstance(results, FormulaSearchResults)
        assert len(results) > 0

        results_octet_pass: list[float] = [
            passes_octet_rule(result.formula) for result in results
        ]

        assert all(results_octet_pass), (
            "Requested octet rule check, but results contain formula"
            " failing octet rule"
        )

        assert any(
            [
                formula_match(
                    test_formula,
                    result.formula,
                ) for result in results
            ]
        ), f"Test formula {test_formula} not found in results"

    def test_empty_results(self):
        """
        Test handling when no valid formulae exist
        """
        results = self.decomposer.decompose(
            query_mass=1.0,  # Too small for any reasonable formula
            ppm_error=1.0,
            min_counts={"C": 1, "H": 1, "O": 1}
        )
        print(
            f"results: {results}"
        )

        assert len(results) == 0

    def test_error_sorting(self):
        """
        Test that FormulaFinder results are sorted by error.
        """
        # Use actual CO2 mass from molmass
        co2_mass = Formula('CO2').monoisotopic_mass

        # Use FormulaFinder for sorted results with error metrics
        finder = FormulaFinder('CHNOPS')
        results = finder.find_formulae(
            mass=co2_mass,
            error_ppm=100.0,
            max_results=10
        )

        assert isinstance(results, FormulaSearchResults)
        if len(results) > 1:
            errors = [abs(result.error_ppm) for result in results]
            assert errors == sorted(errors), "Results should be sorted by error"

class TestFormulaSearchResults:
    pass
    # TODO
