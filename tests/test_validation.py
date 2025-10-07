"""
Tests for formula validation/filtering functions
"""
from molmass import Formula

from find_mfs import passes_octet_rule, get_rdbe


class TestOctetRule:
    """
    Test octet rule validation
    """
    def test_neutral_molecules(self):
        """
        Test neutral molecules, should have integer RDBE
        """
        neutral_molecules = [
            Formula(x) for x in [
                "C6H6",   # Benzene
                "CH4",    # Methane
                "C2H4",   # Ethene
            ]
        ]

        for molecule in neutral_molecules:
            assert passes_octet_rule(molecule)

    def test_odd_charged_molecules(self):
        """
        Test odd-charge molecules, should have half-integer RDBE
        """
        odd_charge_molecules = [
            Formula(x) for x in [
                "C6H13O6+",         # Glucose H+
                "C16H18N2O4SNa+",   # Benzylpenicillin Na+
                "C6H14N4O2H3+++",   # Arginine 3H+++
                "C6H11O6-",         # Glucose -H
                "C10H15N5O13P3-"    # ATP -H
            ]
        ]

        for molecule in odd_charge_molecules:
            assert passes_octet_rule(molecule)

    def test_even_charged_molecules(self):
        """
        Test even-charge molecules, should have integer RDBE
        """
        even_charge_molecules = [
            Formula(x) for x in [
                "C6H14N4O2H2++",        # Arginine 2H++
                "C66H75Cl2N9O24Na2++",  # Vancomycin 2Na++
                "C66H75Cl2N9O24Cl2--",  # Vancomycin 2Cl--
                "C10H14N5O13P3--"       # ATP -2H
            ]
        ]

        for molecule in even_charge_molecules:
            assert passes_octet_rule(molecule)

    def test_invalid_molecules(self):
        """
        These molecules should all fail the octet rule
        """
        invalid_molecules = [
            Formula(x) for x in [
                "C6H6+",
                "CH3",
                "C2H4-",
            ]
        ]

        for molecule in invalid_molecules:
            assert not passes_octet_rule(molecule)


class TestRDBE:
    """
    Test RDBE calculations
    """
    def test_rdbe_calculations(self):
        test_cases = [
            (Formula("CH4"), 0),      # Methane
            (Formula("C2H4"), 1),     # Ethene
            (Formula("C6H6"), 4),     # Benzene
            (Formula("C2H6O"), 0),    # Ethanol
            (Formula("CH2O"), 1),     # Formaldehyde
            (Formula("C5H5N"), 4),    # Pyridine
            (Formula("CH2Cl2"), 0),   # Dichloromethane
            (Formula(
                "C45H69N5O8S"
            ), 14),                   # Apratoxin A
            (Formula(
                "C27H25Br4N3O8"
            ), 15),                   # Psammaplysin E
            (Formula(
                "C10H16N5O13P3"       # ATP
            ), 10)
        ]

        for formula, expected_rdbe in test_cases:
            calculated_rdbe = get_rdbe(formula)
            assert calculated_rdbe == expected_rdbe, (
                f"{formula}: expected RDBE {expected_rdbe}, got {calculated_rdbe}"
            )
