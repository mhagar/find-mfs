"""
This module contains functions for checking molecular formulae against
Senior's theorem, and the octet rule
"""
from typing import Optional
from molmass import Formula, CompositionItem, ELEMENTS

BOND_ELECTRONS: dict[str, int] = {
    'H': 1,
    'Li': 1,
    'Na': 1,
    'C': 4,
    'N': 3,
    'O': 2,
    'F': 1,
    'Cl': 1,
    'Br': 1,
    'I': 1,
    'S': 2,
    'P': 5,
    'Si': 4,
    'B': 3,
}

def filter_formulae(
    formulae: list[Formula] | Formula,
    filter_rdbes: Optional[tuple[float ,float]] = None,
    check_octet: bool = False,
) -> list[Formula]:
    """
    Given Formula object, or a list of Formula objects,
    returns a subset which passes the checks requested by the user
    """
    if isinstance(formulae, Formula):
        # User passed in a single Formula
        formulae = [formulae]

    valid: list[Formula] = []
    for formula in formulae:
        if _filter_formula(
            formula=formula,
            filter_rdbes=filter_rdbes,
            check_octet=check_octet,
        ):
            valid.append(formula)

    return valid

def _filter_formula(
    formula: Formula,
    filter_rdbes: Optional[tuple[float ,float]] = None,
    check_octet: bool = False,
) -> bool:
    # Check RDBEs within limits
    if filter_rdbes is not None:
        calcd_rdbes = get_rdbe(formula)
        min_rdbes, max_rdbes = filter_rdbes
        if not (min_rdbes < calcd_rdbes < max_rdbes):
            return False

    # Check octet rule
    if check_octet:
        if not passes_octet_rule(formula):
            return False

    # All requested checks passed
    return True

def passes_octet_rule(
    formula: Formula
) -> bool:
    """
    Check if a molecular formula satisfies the octet rule.

    Args:
        formula: Molecular formula object

    Returns:
        True if formula satisfies octet rule, False otherwise
    """
    # Calculate RDBE
    rdbe = get_rdbe(formula)

    # Return true/false depending on charge and whether rdbe is integer
    charge = formula.charge
    if abs(charge) % 2.0 == 0.0:
        # charge is even; rdbe should be integer
        return not _is_half_integer(rdbe)

    elif abs(charge) % 2.0 == 1.0:
        # charge is odd; rdbe should be half-integer
        return _is_half_integer(rdbe)

    else:
        raise ValueError(f"Invalid charge: {charge}")

def get_rdbe(
        formula: Formula
) -> Optional[float]:
    """
    Calculate Ring and Double Bond Equivalents (RDBE) for a molecular formula,

    ***NOTE***: This assumes no funny business is going on! i.e.
    no sulfoxides/sulfones, phosphine stuff, radicals, etc.
    This calculation should not be used in those cases.

    $$
    RDBE  = 0.5 x {\sum\limits_{i} n_{i}(b_{i} - 2)} +1
    $$

    where n_i is the number of atoms with number of bond electrons b_i.

    See:
    A Novel Formalism To Characterize the Degree of Unsaturation of
    Organic Molecules. Badertscher, M. et al. (2001)
    doi: 10.1021/ci000135o

    Args:
        formula: molmass.Formula instance

    Returns:
        RDBE value as float, or None if formula contains unhandled element
    """
    n_b_sub_2: list[int] = []
    for element in formula.composition().values():
        element: CompositionItem

        if element.symbol == 'e-':
            continue

        count = element.count
        num_bond_eles = get_bond_electrons(element.symbol)

        if not num_bond_eles:
            return None

        n_b_sub_2.append(
            count * (num_bond_eles - 2)
        )

    return (0.5 * sum(n_b_sub_2)) + 1

def get_bond_electrons(
    symbol: str,
) -> Optional[int]:
    return BOND_ELECTRONS.get(symbol, None)

def _is_half_integer(
    x: float
) -> bool:
    return (2*x) % 2 == 1


