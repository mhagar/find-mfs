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
import find_mfs

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

class TestFindCHNOPSConvenience:
    """
    Test the find_chnops() convenience function
    """

    def setup_method(self):
        """
        Reset the singleton before each test
        """
        find_mfs._default_chnops_finder = None

    def test_basic_functionality(self):
        """
        Test that find_chnops() works for basic queries
        """
        # Test with a known formula: glucose C6H12O6
        glucose_mass = Formula('C6H12O6H+').monoisotopic_mass

        results = find_mfs.find_chnops(
            mass=glucose_mass,
            error_ppm=5.0,
            charge=1,
        )

        assert isinstance(results, FormulaSearchResults)
        assert len(results) > 0
        assert any(
            formula_match(Formula('C6H12O6H+'), result.formula)
            for result in results
        )

    def test_singleton_reuse(self):
        """
        Test that the singleton FormulaFinder is cached and reused across
        calls
        """
        # First call should create the finder
        results1 = find_mfs.find_chnops(mass=180.063, error_ppm=5.0)
        finder1 = find_mfs._default_chnops_finder

        # Second call should reuse the same finder
        results2 = find_mfs.find_chnops(mass=200.047, error_ppm=5.0)
        finder2 = find_mfs._default_chnops_finder

        # Should be the exact same object
        assert finder1 is finder2
        assert isinstance(results1, FormulaSearchResults)
        assert isinstance(results2, FormulaSearchResults)

    def test_with_rdbe_filter(self):
        """
        Test that RDBE filtering works through find_chnops()
        """
        benzene_mass = Formula('C6H6').monoisotopic_mass

        results = find_mfs.find_chnops(
            mass=benzene_mass,
            error_ppm=100.0,
            filter_rdbe=(2, 10)
        )

        assert isinstance(results, FormulaSearchResults)
        assert len(results) > 0

        # Check that all results have RDBE in range
        for result in results:
            assert 2 <= result.rdbe <= 10

    def test_with_octet_check(self):
        """
        Test that octet rule checking works through find_chnops()
        """
        test_mass = Formula('C6H7O+').monoisotopic_mass

        results = find_mfs.find_chnops(
            mass=test_mass,
            charge=1,
            error_ppm=100.0,
            check_octet=True
        )

        assert isinstance(results, FormulaSearchResults)
        assert len(results) > 0

        # Check that all results pass octet rule
        for result in results:
            assert passes_octet_rule(result.formula)

    def test_with_element_constraints(self):
        """
        Test that element count constraints work through find_chnops()
        """
        # Specify element constraints as dict
        results_dict_elements = find_mfs.find_chnops(
            mass=100.0,
            error_ppm=100.0,
            min_counts={'C': 5},
            max_counts={'C': 10, 'H': 20, 'S':0, 'P':0}
        )

        # Specify element constraints as formula string
        results_str_elements = find_mfs.find_chnops(
            mass=100.0,
            error_ppm=100.0,
            min_counts='C5',
            max_counts='C10H20N*S0P0'
        )

        for results in [results_dict_elements, results_str_elements]:
            assert isinstance(results, FormulaSearchResults)

            # Check that all results respect constraints
            for result in results:
                composition = {
                    k: v.count for k, v in result.formula.composition().items()
                }
                c_count = composition.get('C', 0)
                h_count = composition.get('H', 0)

                assert c_count >= 5
                assert c_count <= 10
                assert h_count <= 20

                # Should not have any sulfurs or phosphorous
                assert 'P' not in composition.keys()
                assert 'S' not in composition.keys()

    def test_with_adduct(self):
        """
        Test that adduct handling works through find_chnops()
        """
        glucose_m_p_na = Formula('C6H12O6Na+').monoisotopic_mass

        results = find_mfs.find_chnops(
            mass=glucose_m_p_na,
            charge=1,
            error_ppm=5.0,
            adduct='Na'
        )

        assert isinstance(results, FormulaSearchResults)
        assert len(results) > 0

        # Check that results include sodium
        assert any(
            'Na' in result.formula.formula
            for result in results
        )

        # Check that results include glucode_m_p_na
        assert any(
            formula_match(Formula('C6H12O6Na+'), result.formula)
            for result in results
        )


    def test_error_tolerance_da(self):
        """
        Test that Da-based error tolerance works
        """
        results = find_mfs.find_chnops(
            mass=Formula('C10H22O5').monoisotopic_mass,
            error_da=0.01
        )

        assert isinstance(results, FormulaSearchResults)
        assert len(results) > 0
        assert any(
            formula_match(
                Formula('C10H22O5'),
                result.formula
            ) for result in results
        )


class TestFormulaSearchResults:
    pass
    # TODO
