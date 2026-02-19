"""
Tests for using formula strings as min_counts/max_counts constraints
in FormulaFinder.find_formulae() and find_chnops().

This covers the MS/MS fragment searching use case where a parent ion
formula is given as max_counts so that fragment candidates are
constrained to be elemental subsets of the parent.
"""
import time

import pytest
from molmass import Formula

import find_mfs
from find_mfs import FormulaFinder, FormulaSearchResults, find_chnops
from find_mfs.utils.formulae import to_bounds_dict


@pytest.fixture(autouse=True)
def reset_chnops_singleton():
    """Reset the module-level singleton before each test."""
    find_mfs._default_chnops_finder = None
    yield
    find_mfs._default_chnops_finder = None


@pytest.fixture(scope="module")
def finder():
    """Shared FormulaFinder('CHNOPS') instance for the module."""
    return FormulaFinder('CHNOPS')


# ---------------------------------------------------------------------------
# Baseline correctness tests
# ---------------------------------------------------------------------------

class TestNovobiocinFragment:
    """Test 1: Novobiocin fragment — the canonical example."""

    def test_finds_single_result(self, finder):
        results = finder.find_formulae(
            mass=189.09133,
            charge=1,
            error_ppm=5.0,
            max_counts='C31H37N2O11',
            check_octet=True,
            filter_rdbe=(0, 20),
        )
        assert isinstance(results, FormulaSearchResults)
        assert len(results) == 1, (
            f"Expected exactly 1 result, got {len(results)}: "
            f"{[r.formula.formula for r in results]}"
        )

    def test_correct_formula(self, finder):
        results = finder.find_formulae(
            mass=189.09133,
            charge=1,
            error_ppm=5.0,
            max_counts='C31H37N2O11',
            check_octet=True,
            filter_rdbe=(0, 20),
        )
        result = results[0]
        # C12H13O2 with +1 charge
        expected = Formula('[C12H13O2]+')
        assert result.formula.formula == expected.formula, (
            f"Expected {expected.formula}, got {result.formula.formula}"
        )

    def test_error_ppm_value(self, finder):
        results = finder.find_formulae(
            mass=189.09133,
            charge=1,
            error_ppm=5.0,
            max_counts='C31H37N2O11',
            check_octet=True,
            filter_rdbe=(0, 20),
        )
        result = results[0]
        assert abs(result.error_ppm - (-1.71)) < 0.1, (
            f"Expected error_ppm ~ -1.71, got {result.error_ppm:.2f}"
        )


class TestGlucoseFragment:
    """Test 2: Glucose as parent — search for fragments of glucose [M+H]+."""

    def test_water_loss_fragment(self, finder):
        # Glucose [M+H]+: C6H12O6 + H+ = C6H13O6+
        glucose_mh_mass = Formula('[C6H13O6]+').monoisotopic_mass
        # Water loss fragment: C6H11O5+
        water_loss_mass = Formula('[C6H11O5]+').monoisotopic_mass

        results = finder.find_formulae(
            mass=water_loss_mass,
            charge=1,
            error_ppm=5.0,
            max_counts='C6H13O6',
        )
        assert isinstance(results, FormulaSearchResults)
        assert len(results) > 0, "Should find at least one fragment"
        assert any(
            r.formula.formula == Formula('[C6H11O5]+').formula
            for r in results
        ), (
            f"Expected C6H11O5+ among results, got: "
            f"{[r.formula.formula for r in results]}"
        )


class TestSubsetCorrectness:
    """Test 3: For any result from a max_counts-constrained search,
    every element count in the result must be <= the parent count."""

    def test_all_results_are_subsets(self, finder):
        parent = 'C31H37N2O11'
        parent_counts = to_bounds_dict(
            parent, elements=['C', 'H', 'N', 'O', 'P', 'S']
        )

        results = finder.find_formulae(
            mass=189.09133,
            charge=1,
            error_ppm=10.0,
            max_counts=parent,
        )
        assert len(results) > 0

        for result in results:
            composition = {
                k: v.count for k, v in result.formula.composition().items()
            }
            for element, max_count in parent_counts.items():
                actual = composition.get(element, 0)
                assert actual <= max_count, (
                    f"Formula {result.formula.formula}: "
                    f"{element}={actual} > parent {element}={max_count}"
                )


class TestDictStringEquivalence:
    """Test 4: Searching with max_counts as string should give
    identical results to the equivalent dict."""

    def test_sucrose_constraint_equivalence(self, finder):
        mass = 150.0
        string_results = finder.find_formulae(
            mass=mass,
            error_ppm=10.0,
            max_counts='C12H22O11',
        )
        dict_results = finder.find_formulae(
            mass=mass,
            error_ppm=10.0,
            max_counts={'C': 12, 'H': 22, 'O': 11, 'N': 0, 'P': 0, 'S': 0},
        )
        str_formulae = sorted(r.formula.formula for r in string_results)
        dict_formulae = sorted(r.formula.formula for r in dict_results)
        assert str_formulae == dict_formulae, (
            f"String vs dict mismatch:\n"
            f"  string: {str_formulae}\n"
            f"  dict:   {dict_formulae}"
        )


class TestMinCountsAsString:
    """Test 5: min_counts='C5' constrains results to have >= 5 carbons."""

    def test_min_carbons(self, finder):
        results = finder.find_formulae(
            mass=150.0,
            error_ppm=10.0,
            min_counts='C5',
        )
        assert len(results) > 0
        for result in results:
            composition = {
                k: v.count for k, v in result.formula.composition().items()
            }
            c_count = composition.get('C', 0)
            assert c_count >= 5, (
                f"Formula {result.formula.formula} has {c_count} carbons, "
                f"expected >= 5"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestWildcardConstraint:
    """Test 6: max_counts='C20H40O*' should leave O unbounded."""

    def test_wildcard_leaves_element_unbounded(self, finder):
        results = finder.find_formulae(
            mass=300.0,
            error_ppm=10.0,
            max_counts='C20H40O*',
        )
        # O* means oxygen is unbounded — the parsed dict should have O=inf
        parsed = to_bounds_dict(
            'C20H40O*', elements=['C', 'H', 'N', 'O', 'P', 'S']
        )
        assert parsed['O'] == float('inf')
        assert parsed['N'] == 0
        assert parsed['P'] == 0
        assert parsed['S'] == 0

        # Results should have no N, P, S but may have any amount of O
        for result in results:
            composition = {
                k: v.count for k, v in result.formula.composition().items()
            }
            assert composition.get('N', 0) == 0
            assert composition.get('P', 0) == 0
            assert composition.get('S', 0) == 0


class TestZeroCounts:
    """Test 7: max_counts='C20H40P0S0' should exclude P and S."""

    def test_zero_excludes_elements(self, finder):
        # N* and O* leave those unbounded; P0 and S0 forbid them
        results = finder.find_formulae(
            mass=200.0,
            error_ppm=10.0,
            max_counts='C20H40N*O*P0S0',
        )
        assert len(results) > 0
        for result in results:
            composition = {
                k: v.count for k, v in result.formula.composition().items()
            }
            assert composition.get('P', 0) == 0, (
                f"Formula {result.formula.formula} contains P"
            )
            assert composition.get('S', 0) == 0, (
                f"Formula {result.formula.formula} contains S"
            )


class TestTightConstraint:
    """Test 8: When max_counts is a small formula like 'CH4',
    the search space is tiny."""

    def test_small_formula_constraint(self, finder):
        # CH4 monoisotopic mass = ~16.031 Da (neutral)
        ch4_mass = Formula('CH4').monoisotopic_mass
        results = finder.find_formulae(
            mass=ch4_mass,
            error_ppm=5.0,
            max_counts='CH4',
        )
        assert len(results) >= 1
        # The only result should be CH4 itself
        assert any(
            r.formula.formula == Formula('CH4').formula
            for r in results
        )


class TestEmptyResults:
    """Test 9: max_counts set to a formula whose mass is much lower
    than query_mass should yield 0 results."""

    def test_impossible_constraint_gives_empty(self, finder):
        results = finder.find_formulae(
            mass=500.0,
            error_ppm=5.0,
            max_counts='CH4',
        )
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Performance regression guards
# ---------------------------------------------------------------------------

class TestPerformance:
    """Tests 10-11: Performance regression guards for constrained searches."""

    def test_300da_fragment_search_under_200ms(self, finder):
        start = time.perf_counter()
        results = finder.find_formulae(
            mass=300.0,
            charge=1,
            error_ppm=5.0,
            max_counts='C50H80N10O20P5S3',
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 200, (
            f"300 Da constrained search took {elapsed_ms:.0f} ms (limit: 200 ms)"
        )
        assert isinstance(results, FormulaSearchResults)

    def test_500da_fragment_search_under_500ms(self, finder):
        start = time.perf_counter()
        results = finder.find_formulae(
            mass=500.0,
            charge=1,
            error_ppm=5.0,
            max_counts='C50H80N10O20P5S3',
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500, (
            f"500 Da constrained search took {elapsed_ms:.0f} ms (limit: 500 ms)"
        )
        assert isinstance(results, FormulaSearchResults)


# ---------------------------------------------------------------------------
# find_chnops convenience function
# ---------------------------------------------------------------------------

class TestFindChnopsWithStringConstraints:
    """Verify the convenience function also accepts string constraints."""

    def test_find_chnops_with_max_counts_string(self):
        results = find_chnops(
            mass=189.09133,
            charge=1,
            error_ppm=5.0,
            max_counts='C31H37N2O11',
            check_octet=True,
            filter_rdbe=(0, 20),
        )
        assert isinstance(results, FormulaSearchResults)
        assert len(results) == 1
        assert results[0].formula.formula == Formula('[C12H13O2]+').formula

    def test_find_chnops_with_min_counts_string(self):
        results = find_chnops(
            mass=150.0,
            error_ppm=10.0,
            min_counts='C5',
        )
        assert len(results) > 0
        for result in results:
            composition = {
                k: v.count for k, v in result.formula.composition().items()
            }
            assert composition.get('C', 0) >= 5
