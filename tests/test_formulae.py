"""
Tests for formula utility functions
"""
import pytest
from find_mfs.utils.formulae import to_bounds_dict


class TestToBoundsDict:
    """
    Tests for to_bounds_dict() function which parses constraint strings
    """

    def test_basic_constraint(self):
        """
        Test basic constraint string with explicit counts
        """
        result = to_bounds_dict(
            formula="C20H10O5",
            elements=["C", "H", "N", "O", "P", "S"]
        )
        expected = {"C": 20, "H": 10, "O": 5, "N": 0, "P": 0, "S": 0}
        assert result == expected

    def test_parent_ion_constraint(self):
        """
        All unmentioned elements default to 0, as if the user
        was querying a fragment of a parent ion
        """
        result = to_bounds_dict(
            formula="C12H22O11",
            elements=["C", "H", "N", "O", "P", "S"]
        )
        expected = {"C": 12, "H": 22, "O": 11, "N": 0, "P": 0, "S": 0}
        assert result == expected

    def test_element_without_count_defaults_to_1(self):
        """
        Test that 'S' is interpreted as 'S1'
        """
        result = to_bounds_dict(
            formula="C12H22O11S", elements=["C", "H", "N", "O", "P", "S"])
        expected = {"C": 12, "H": 22, "O": 11, "S": 1, "N": 0, "P": 0}
        assert result == expected

    def test_explicit_zero_count(self):
        """
        Test explicit zero count like 'P0' to forbid an element
        """
        result = to_bounds_dict(
            formula="C20H10O5P0",
            elements=["C", "H", "N", "O", "P", "S"],
        )
        expected = {"C": 20, "H": 10, "O": 5, "P": 0, "N": 0, "S": 0}
        assert result == expected

    def test_multiple_explicit_zeros(self):
        """
        Test multiple explicit zero counts
        """
        result = to_bounds_dict(
            formula="C20H10O5P0S0N0",
            elements=["C", "H", "N", "O", "P", "S"]
        )
        expected = {"C": 20, "H": 10, "O": 5, "P": 0, "S": 0, "N": 0}
        assert result == expected

    def test_two_letter_elements(self):
        """
        Test elements with two-letter symbols like Cl, Br
        """
        result = to_bounds_dict(formula="C6H5Cl", elements=["C", "H", "Cl"])
        expected = {"C": 6, "H": 5, "Cl": 1}
        assert result == expected

    def test_two_letter_elements_with_counts(self):
        """
        Test two-letter elements with explicit counts
        """
        result = to_bounds_dict(formula="C6H5Cl2Br3", elements=["C", "H", "Cl", "Br"])
        expected = {"C": 6, "H": 5, "Cl": 2, "Br": 3}
        assert result == expected

    def test_element_not_in_allowed_set_raises_error(self):
        """
        Test that using an element not in the allowed set raises ValueError
        """
        with pytest.raises(ValueError, match="not in the given element set"):
            to_bounds_dict(formula=
                "C12H22O11S",
                elements=["C", "H", "N", "O", "P"], # S not allowed
            )

    def test_invalid_element_symbol_raises_error(self):
        """
        Test that invalid element symbols raise ValueError
        """
        with pytest.raises(ValueError, match="Invalid element symbol"):
            # "Zz" is not a valid element symbol in periodic table
            to_bounds_dict(formula=
                "C12H22Zz5",
                elements=["C", "H", "N", "O", "P", "S"],
            )

    def test_empty_string(self):
        """
        Test empty constraint string - all elements should be 0
        """
        result = to_bounds_dict(formula="", elements=["C", "H", "N", "O", "P", "S"])
        expected = {"C": 0, "H": 0, "N": 0, "O": 0, "P": 0, "S": 0}
        assert result == expected

    def test_single_element_with_count(self):
        """
        Test single element constraint
        """
        result = to_bounds_dict(formula="C20", elements=["C", "H", "N", "O", "P", "S"])
        expected = {"C": 20, "H": 0, "N": 0, "O": 0, "P": 0, "S": 0}
        assert result == expected

    def test_single_element_without_count(self):
        """
        Test single element without count defaults to 1
        """
        result = to_bounds_dict(formula="C", elements=["C", "H", "N", "O", "P", "S"])
        expected = {"C": 1, "H": 0, "N": 0, "O": 0, "P": 0, "S": 0}
        assert result == expected

    def test_large_counts(self):
        """
        Test with large element counts
        """
        result = to_bounds_dict(formula="C100H200O50N25", elements=["C", "H", "N", "O"])
        expected = {"C": 100, "H": 200, "O": 50, "N": 25}
        assert result == expected

    def test_minimal_element_set(self):
        """
        Test with minimal element set
        """
        result = to_bounds_dict(formula="CH4", elements=["C", "H"])
        expected = {"C": 1, "H": 4}
        assert result == expected

    def test_chnops_default(self):
        """
        Test typical CHNOPS usage
        """
        elements = ["C", "H", "N", "O", "P", "S"]

        # Full specification
        result = to_bounds_dict(
            formula="C20H40N5O10P2S",
            elements=elements,
        )
        expected = {"C": 20, "H": 40, "N": 5, "O": 10, "P": 2, "S": 1}
        assert result == expected

    def test_order_independence(self):
        """
        Test that element order in string doesn't matter
        """
        elements = ["C", "H", "N", "O", "P", "S"]

        # Different orders should give same result
        result1 = to_bounds_dict(
            formula="C20H10O5",
            elements=elements
        )
        result2 = to_bounds_dict(
            formula="H10O5C20",
            elements=elements
        )
        result3 = to_bounds_dict(
            formula="O5C20H10",
            elements=elements
        )

        assert result1 == result2 == result3
